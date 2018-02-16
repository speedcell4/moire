import dynet as dy

from moire import nn, Expression, ParameterCollection

__all__ = [
    'ConjugateEmbedding',
]


class ConjugateEmbedding(nn.Module):
    def __init__(self, pc: ParameterCollection, num_embeddings: int,
                 embedding_dim_fixed: int, embedding_dim_training: int):
        super(ConjugateEmbedding, self).__init__(pc)

        self.num_embeddings = num_embeddings
        self.embedding_dim_fixed = embedding_dim_fixed
        self.embedding_dim_training = embedding_dim_training
        self.embedding_dim = embedding_dim_fixed + embedding_dim_training

        self.embedding_fixed = self.pc.add_lookup_parameters((num_embeddings, embedding_dim_fixed))
        self.embedding_training = self.pc.add_lookup_parameters((num_embeddings, embedding_dim_training))

    def __repr__(self):
        return f'{self.__class__.__name__} ({self.num_embeddings} tokens, {self.embedding_dim} dim)'

    def __call__(self, ix: int) -> Expression:
        f = dy.lookup(self.embedding_fixed, ix, update=False)
        t = dy.lookup(self.embedding_training, ix, update=self.training)
        return dy.concatenate([f, t])


if __name__ == '__main__':
    embedding = ConjugateEmbedding(ParameterCollection(), 100, 2, 3)
    dy.renew_cg(True, True)

    print(embedding(2).dim())
