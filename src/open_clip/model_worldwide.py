from src.open_clip.model import CLIP


class WorldWideCLIP(CLIP):
    def encode_text(self, text):
        x = self.token_embedding(text)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, attn_mask=self.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        # x.shape = [batch_size, n_ctx, transformer.width]
        index = (text > 1).sum(-1)  # eos is 1 for worldwide (XLM-V) tokenizer.
        
        x = x[torch.arange(x.shape[0]), index] @ self.text_projection
        return x
