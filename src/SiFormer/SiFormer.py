import torch
import torch.nn as nn

NUM_HAND_JOINTS = 21
NUM_COORDS = 3
import SiFormer.FeatureIsolatedTransformer as FeatureIsolatedTransformer


class SiFormer(nn.Module):
    """
    SiFormer (Sign Isolated Transformer) model for sign language recognition.
    Uses left and right hand landmarks as input features.
    """
    def __init__(self, num_classes=100, num_hid=128, num_enc_layers=3, num_dec_layers=2,
                 patience=1, seq_len=50, device=None):
        super(SiFormer, self).__init__()
        print("Initializing SiFormer model")

        self.seq_len = seq_len
        self.device = device
        self.num_hid = num_hid

        # Positional encodings for left/right hands
        self.l_hand_embedding = nn.Parameter(self.get_encoding_table(NUM_HAND_JOINTS * NUM_COORDS))
        self.r_hand_embedding = nn.Parameter(self.get_encoding_table(NUM_HAND_JOINTS * NUM_COORDS))

        # Classification query (1 token)
        self.class_query = nn.Parameter(torch.randn(1, 1, num_hid))  # shape: [1, 1, d_model]

        # FeatureIsolatedTransformer with 2 streams (left, right hand)
        self.transformer = FeatureIsolatedTransformer(
            feature_dims=[NUM_HAND_JOINTS * NUM_COORDS] * 2,  # [63, 63]
            num_layers=[num_enc_layers] * 2,                   # same for both hands
            num_decoder_layers=num_dec_layers,
            inner_classifiers_config=[num_hid, num_classes],
            projections_config=[seq_len, 1],
            device=device,
            patience=patience
        )

        print(f"Encoder layers: {num_enc_layers}, Decoder layers: {num_dec_layers}, Patience: {patience}")
        self.projection = nn.Linear(num_hid, num_classes)

    def forward(self, l_hand, r_hand, training=True):
        #batch_size, seq_len, num_joints, coord_dim = l_hand.shape  # [B, T, 21, 3]
        batch_size, seq_len, feature_dim = l_hand.shape  # [B, T, 63]

        # Flatten joints
        l_hand_flat = l_hand.reshape(batch_size, seq_len, -1)      # [B, T, 63]
        r_hand_flat = r_hand.reshape(batch_size, seq_len, -1)      # [B, T, 63]

        # To shape: [T, B, C]
        l_input = l_hand_flat.permute(1, 0, 2).float() + self.l_hand_embedding  # [T, B, 63]
        r_input = r_hand_flat.permute(1, 0, 2).float() + self.r_hand_embedding  # [T, B, 63]

        # Classification token
        tgt = self.class_query.repeat(1, batch_size, 1)  # [1, B, D]

        # Transformer
        output = self.transformer([l_input, r_input], tgt, training=training)  # [1, B, D]
        output = output.transpose(0, 1)  # [B, 1, D]

        logits = self.projection(output.squeeze(1))  # [B, num_classes]
        return logits

    @staticmethod
    def get_encoding_table(d_model=63, seq_len=50):
        """Positional encoding: random & monotonic for compatibility."""
        torch.manual_seed(42)
        table = torch.rand(seq_len, d_model)
        for i in range(seq_len):
            for j in range(1, d_model):
                table[i, j] = table[i, j - 1]  # make monotonic
        return table.unsqueeze(1)  # [T, 1, D]
