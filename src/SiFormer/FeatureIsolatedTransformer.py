
NUM_HAND_JOINTS = 21
NUM_COORDS = 3
import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureIsolatedTransformer(nn.Module):
    def __init__(self, feature_dims, num_layers, num_decoder_layers=2,
                 inner_classifiers_config=None, projections_config=None, device=None, patience=1):
        super(FeatureIsolatedTransformer, self).__init__()

        self.d_model = sum(feature_dims)
        self.device = device
        self.patience = patience
        self.inner_classifiers = None
        self.projections = None
        self.dropout = 0.1
        self.activation = F.relu

        # Per-feature encoders
        self.encoder_by_feature = nn.ModuleList()
        for i, feature_dim in enumerate(feature_dims):
            nhead = max([h for h in range(1, 9) if feature_dim % h == 0])
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=feature_dim,
                nhead=nhead,
                dim_feedforward=feature_dim * 4,
                dropout=self.dropout,
                activation=self.activation
            )
            self.encoder_by_feature.append(nn.TransformerEncoder(encoder_layer, num_layers[i]))

        # Inner classifiers (for early exit)
        if inner_classifiers_config:
            d_model, num_classes = inner_classifiers_config
            self.inner_classifiers = nn.ModuleList([
                nn.Linear(d_model, num_classes) for _ in range(num_decoder_layers)
            ])

        # Optional projections
        if projections_config:
            seq_len, _ = projections_config
            self.projections = nn.ModuleList([
                nn.Linear(self.d_model, seq_len) for _ in range(num_decoder_layers)
            ])

        # Decoder
        decoder_nhead = max([h for h in range(1, 9) if self.d_model % h == 0])
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.d_model,
            nhead=decoder_nhead,
            dim_feedforward=self.d_model * 4,
            dropout=self.dropout,
            activation=self.activation
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)

        self.norm = nn.LayerNorm(self.d_model)

    def forward(self, sources, tgt, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None, training=True):

        # Encode per feature (left, right)
        encoded_features = [encoder(src) for encoder, src in zip(self.encoder_by_feature, sources)]
        memory = torch.cat(encoded_features, dim=2)

        # Decode with optional early exit
        output = tgt
        if training and self.inner_classifiers:
            patience_counter = 0
            last_result = None

            for i, decoder_layer in enumerate(self.decoder.layers):
                output = decoder_layer(output, memory)

                normed = self.norm(output)
                logits = self.inner_classifiers[i](normed)

                pred = torch.argmax(logits, dim=2)
                if last_result is not None and torch.equal(pred, last_result):
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        break
                else:
                    patience_counter = 0
                last_result = pred
        else:
            output = self.decoder(tgt, memory)

        return self.norm(output)


