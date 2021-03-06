import dynet as dy

import moire
from moire import nn, Expression, ParameterCollection
from moire.nn.initializers import Uniform

__all__ = [
    'Embedding',
]


class Embedding(nn.Module):
    def __init__(self, pc: ParameterCollection, num_embeddings: int, embedding_dim: int, initializer=Uniform()) -> None:
        super(Embedding, self).__init__(pc)

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        self.embedding = self.add_lookup((num_embeddings, embedding_dim), initializer)

    def __repr__(self):
        return f'{self.__class__.__name__} ({self.num_embeddings} tokens, {self.embedding_dim} dim)'

    def __call__(self, ix: int) -> Expression:
        return dy.lookup(self.embedding, ix, update=moire.config.train)


if __name__ == '__main__':
    embedding = Embedding(ParameterCollection(), 100, 10)
    dy.renew_cg(True, True)

    moire.debug(embedding(2).dim())
