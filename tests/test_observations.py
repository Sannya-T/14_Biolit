import polars as pl
from polars import col

from biolit.observations import learnable_taxonomy


class TestLearnableTaxonomy:
    def test_valid_deepest_taxon(self):
        inp = pl.DataFrame(
            {
                "nom_scientifique": ["herbe", "herbe"],
                "genre": ["plante", "plante"],
                "classe": ["chlorophyle", "chlorophyle"],
                "n_obs": 1,
            }
        )

        out = learnable_taxonomy(inp, ["genre", "classe"], 2)
        exp = ["herbe"]
        assert out == exp
