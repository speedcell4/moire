import dynet as dy

import moire
from moire import Expression, ParameterCollection, nn
from moire.nn.initializers import Uniform

__all__ = [
    'ConjugateEmbedding',
]


class ConjugateEmbedding(nn.Module):
    def __init__(self, pc: ParameterCollection, num_embeddings: int,
                 embedding_dim_fixed: int, embedding_dim_training: int, initializer=Uniform()) -> None:
        super(ConjugateEmbedding, self).__init__(pc)

        self.num_embeddings = num_embeddings
        self.embedding_dim_fixed = embedding_dim_fixed
        self.embedding_dim_training = embedding_dim_training
        self.embedding_dim = embedding_dim_fixed + embedding_dim_training

        self.embedding_fixed = self.add_lookup((num_embeddings, embedding_dim_fixed), initializer)
        self.embedding_training = self.add_lookup((num_embeddings, embedding_dim_training), initializer)

    def __repr__(self):
        return f'{self.__class__.__name__} ({self.num_embeddings} tokens, {self.embedding_dim} dim)'

    def __call__(self, ix: int) -> Expression:
        f = dy.lookup(self.embedding_fixed, ix, update=False)
        t = dy.lookup(self.embedding_training, ix, update=moire.config.train)
        return dy.concatenate([f, t])


if __name__ == '__main__':
    embedding = ConjugateEmbedding(ParameterCollection(), 100, 2, 3)
    dy.renew_cg(True, True)

    moire.debug(embedding(2).dim())
