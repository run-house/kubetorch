from typing import Any, Dict

from sphinx.application import Sphinx
from sphinx.environment.adapters.toctree import TocTree
from sphinxcontrib.serializinghtml import JSONHTMLBuilder

__version__ = "0.0.1"


def setup(app: Sphinx) -> Dict[str, Any]:
    app.add_builder(SphinxGlobalTOCJSONHTMLBuilder, override=True)

    return {"version": __version__, "parallel_read_safe": True}


class SphinxGlobalTOCJSONHTMLBuilder(JSONHTMLBuilder):

    name: str = "json"

    def get_doc_context(self, docname: str, body: str, metatags: str) -> Dict[str, Any]:
        """
        Extends :py:class:`sphinxcontrib.serializinghtml.JSONHTMLBuilder`.

        Add a ``globaltoc`` key to our document that contains the HTML for the
        global table of contents.

        Note:

            We're rendering the **full global toc** for the entire documentation
            set into every page. We do this to easily render the toc on each
            page and allow for a unique toc for each branch and repo version.
        """
        doc = super().get_doc_context(docname, body, metatags)
        # Get the entire doctree.  It is the 3rd argument (``collapse``) that
        # does this.  If you set that to ``True`` you will only get the submenu
        # HTML included if you are on a page that is within that submenu.
        self_toctree = TocTree(self.env).get_toctree_for(
            "index", self, False, titles_only=True, includehidden=False, maxdepth=2
        )
        toctree = self.render_partial(self_toctree)["fragment"]
        doc["globaltoc"] = toctree
        return doc
