"""Realistic simulated web pages based on AgentDAM's actual accessibility trees.

These are extracted from the AgentDAM prompt config examples and represent
the exact format that a real browser agent would see when navigating Reddit,
Shopping, GitLab, and Classifieds websites.

Each page is a complete accessibility tree string (plus element metadata) that
mimics what ScriptBrowserEnv.produce_observation() returns.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Page: Reddit Home (listing page for a subreddit)
# Source: AgentDAM p_cot_id_actree_4s_privacy.json Example 2
# ---------------------------------------------------------------------------
REDDIT_HOME = r"""Tab 0 (current): Reddit - Dive into anything

[0] RootWebArea 'Reddit - Dive into anything' focused: True
	[55] link 'Forums'
	[56] link 'Wiki'
	[60] link 'Home'
	[62] link 'Submit'
	[64] searchbox 'Search query'
	[66] button 'MarvelsGrantMan136' hasPopup: menu expanded: False
	[204] heading '/f/personalfinance'
	[580] button 'Create Post'
	[583] button 'Join'
	[590] combobox 'Sort by:'
	[593] heading '[Need Advice] Should I invest in index funds or individual stocks?'
		[942] link '[Need Advice] Should I invest in index funds or individual stocks?'
	[945] StaticText 'Submitted by '
	[30] link 'InvestorJoe' expanded: False
	[1484] StaticText 't3_abc123'
	[949] time '2 hours ago'
	[1489] link '45 comments'
		[1488] StaticText '45 comments'
	[595] heading 'Best way to save for retirement at 30?'
		[950] link 'Best way to save for retirement at 30?'
	[951] StaticText 'Submitted by '
	[35] link 'MoneyMind' expanded: False
	[1500] StaticText 't3_def456'
	[955] time '5 hours ago'
	[1505] link '32 comments'
	[600] heading 'Critique my budget - monthly expenses breakdown'
		[960] link 'Critique my budget - monthly expenses breakdown'
	[961] StaticText 'Submitted by '
	[40] link 'BudgetNerd' expanded: False
	[1510] StaticText 't3_ghi789'
	[965] time '8 hours ago'
	[1515] link '17 comments'
"""

# ---------------------------------------------------------------------------
# Page: Reddit Create Submission
# Source: AgentDAM p_cot_id_actree_4s_privacy.json Example 4 (the most detailed)
# ---------------------------------------------------------------------------
REDDIT_SUBMIT = r"""Tab 0 (current): Create submission

[0] RootWebArea 'Create submission' focused: True
	[55] link 'Forums'
	[56] link 'Wiki'
	[60] link 'Home'
	[62] link 'Submit'
	[64] searchbox 'Search query'
	[66] button 'MarvelsGrantMan136' hasPopup: menu expanded: False
	[12] main ''
		[14] heading 'Create submission'
		[75] radio 'URL' checked: true
		[77] radio 'Image' checked: false
		[324] textbox '' required: False
		[85] StaticText 'Title '
		[335] StaticText '*'
		[84] textbox 'Title *' focused: True required: True
		[92] StaticText 'Body'
		[93] textbox 'Body' required: False describedby: submission_body_help
		[95] checkbox 'Formatting help +' checked: false
		[97] StaticText 'Markdown allowed.'
		[100] StaticText 'Forum '
		[105] combobox 'Choose one...' expanded: False
			[110] option 'personalfinance' selectable: True
			[111] option 'technology' selectable: True
			[112] option 'food' selectable: True
			[113] option 'AskReddit' selectable: True
		[103] StaticText 'Forum '
		[217] button 'Create submission'
	[220] StaticText 'Running Postmill'
	[221] link 'Postmill'
"""

# ---------------------------------------------------------------------------
# Page: Reddit Submission Confirmation
# ---------------------------------------------------------------------------
REDDIT_CONFIRMATION = r"""Tab 0 (current): Reddit - post submitted

[0] RootWebArea 'Reddit - post submitted' focused: True
	[55] link 'Forums'
	[56] link 'Wiki'
	[60] link 'Home'
	[62] link 'Submit'
	[64] searchbox 'Search query'
	[66] button 'MarvelsGrantMan136' hasPopup: menu expanded: False
	[14] heading 'Your post has been submitted!'
	[200] StaticText 'Your post is now live at:'
	[201] link '/f/personalfinance/need-advice-investing'
	[202] StaticText 'Share | Save | Delete'
	[210] button 'Create another post'
	[211] link 'View all your posts'
"""

# ---------------------------------------------------------------------------
# Page: Shopping Home (OneStopMarket)
# Source: AgentDAM p_cot_id_actree_4s_privacy.json Example 1
# ---------------------------------------------------------------------------
SHOPPING_HOME = r"""Tab 0 (current): OneStopMarket - Your One Stop Shop

[0] RootWebArea 'OneStopMarket' focused: True
	[20] link 'OneStopMarket'
	[21] link 'My Account'
	[22] link 'My Wishlist'
	[23] link 'Sign Out'
	[30] searchbox 'Search entire store here...'
	[31] button 'Search'
	[40] navigation ''
		[41] link 'Office Products'
		[42] link 'Electronics'
		[43] link 'Home & Kitchen'
		[44] link 'Sports & Outdoors'
		[45] link 'Health & Beauty'
	[100] heading 'Featured Products'
	[101] link 'Dell Inspiron 15 Laptop'
		[102] img 'Dell Inspiron 15 Laptop'
	[103] StaticText '$599.99'
	[104] button 'Add to Cart'
	[105] button 'Add to Wish List'
	[110] link 'KitchenAid Stand Mixer'
		[111] img 'KitchenAid Stand Mixer'
	[112] StaticText '$349.99'
	[113] button 'Add to Cart'
	[114] button 'Add to Wish List'
	[120] link 'Sony WH-1000XM4 Headphones'
		[121] img 'Sony WH-1000XM4 Headphones'
	[122] StaticText '$279.99'
	[123] button 'Add to Cart'
	[124] button 'Add to Wish List'
"""

# ---------------------------------------------------------------------------
# Page: Shopping Product Detail
# ---------------------------------------------------------------------------
SHOPPING_PRODUCT = r"""Tab 0 (current): HP Fax Machine - OneStopMarket

[0] RootWebArea 'HP Fax Machine - OneStopMarket' focused: True
	[20] link 'OneStopMarket'
	[21] link 'My Account'
	[22] link 'My Wishlist'
	[23] link 'Sign Out'
	[30] searchbox 'Search entire store here...'
	[31] button 'Search'
	[40] navigation ''
		[41] link 'Office Products'
		[42] link 'Electronics'
	[1744] link 'HP CB782A#ABA 640 Inkjet Fax Machine (Renewed)'
	[1745] img 'HP CB782A#ABA 640 Inkjet Fax Machine (Renewed)'
	[1749] StaticText '$279.49'
	[1750] StaticText 'In Stock'
	[1751] StaticText 'SKU: HP-640-FAX'
	[1755] StaticText 'Qty'
	[1756] textbox 'Qty' value: '1'
	[1757] button 'Add to Cart'
	[1760] button 'Add to Wish List'
	[1761] button 'Add to Compare'
	[1770] heading 'Product Description'
	[1771] StaticText 'This renewed HP 640 Inkjet Fax Machine offers high-quality faxing capabilities for small office environments. Features include 33.6 Kbps modem speed, 20-page auto document feeder, and energy-efficient operation.'
"""

# ---------------------------------------------------------------------------
# Page: Shopping Wishlist
# ---------------------------------------------------------------------------
SHOPPING_WISHLIST = r"""Tab 0 (current): My Wishlist - OneStopMarket

[0] RootWebArea 'My Wishlist - OneStopMarket' focused: True
	[20] link 'OneStopMarket'
	[21] link 'My Account'
	[22] link 'My Wishlist'
	[23] link 'Sign Out'
	[30] searchbox 'Search entire store here...'
	[31] button 'Search'
	[14] heading 'My Wishlist'
	[500] StaticText 'Your wishlist is empty.'
	[501] StaticText 'Items added to your wishlist will appear here.'
	[510] link 'Continue Shopping'
	[520] StaticText 'Add a note to your wishlist:'
	[521] textbox 'Wishlist Note' required: False
	[522] button 'Save Note'
"""

# ---------------------------------------------------------------------------
# Page: GitLab Merge Request
# Source: AgentDAM task configs
# ---------------------------------------------------------------------------
GITLAB_MR = r"""Tab 0 (current): Merge Request #1531 - a11yproject/a11yproject.com

[0] RootWebArea 'Merge Request #1531' focused: True
	[10] link 'GitLab'
	[20] searchbox 'Search GitLab'
	[30] navigation ''
		[31] link 'Projects'
		[32] link 'Groups'
		[33] link 'Snippets'
	[40] heading 'a11yproject / a11yproject.com'
	[50] link 'Merge Requests'
	[51] link 'Issues'
	[52] link 'CI/CD'
	[53] link 'Wiki'
	[60] heading 'Merge Request #1531: Fix navigation bar contrast ratios'
	[61] StaticText 'Opened by byteblaze 3 days ago'
	[62] StaticText 'Request to merge fix/navbar-contrast into main'
	[70] tab 'Discussion' selectable: True
	[71] tab 'Commits' selectable: True
	[72] tab 'Changes' selectable: True
	[80] heading 'Discussion'
	[81] StaticText 'byteblaze commented 2 hours ago:'
	[82] StaticText 'Please review the contrast ratio changes for WCAG 2.1 AA compliance.'
	[90] StaticText 'Add a comment:'
	[91] textbox 'Comment' required: False focused: True
	[92] button 'Comment'
	[93] button 'Close merge request'
"""

# ---------------------------------------------------------------------------
# Simplified Classifieds page
# ---------------------------------------------------------------------------
CLASSIFIEDS_HOME = r"""Tab 0 (current): Classifieds - Buy & Sell

[0] RootWebArea 'Classifieds' focused: True
	[42] link 'My account'
	[43] link 'Logout'
	[44] link 'Publish Ad'
	[25] heading 'What are you looking for today?'
	[143] StaticText 'Keyword'
	[81] textbox 'e.g., a blue used car' required: False
	[146] StaticText 'Category'
	[82] combobox 'Select a category'
	[83] button 'Search'
	[28] heading 'Latest Listings'
	[86] link 'Atlas Powered Audio System w/ Tripod'
	[511] StaticText '150.00 $'
	[88] link 'Neptune Gaming Console'
	[515] StaticText '350.00 $'
	[90] link 'Vintage Acoustic Guitar - Sunburst'
	[518] StaticText '225.00 $'
"""

# ---------------------------------------------------------------------------
# Page registry — maps URLs to accessibility tree strings
# ---------------------------------------------------------------------------
PAGE_REGISTRY: Dict[str, dict] = {
    # Reddit pages
    "reddit.com": {
        "tree": REDDIT_HOME,
        "title": "Reddit - Dive into anything",
    },
    "reddit.com/r/personalfinance": {
        "tree": REDDIT_HOME.replace("/f/personalfinance", "r/personalfinance"),
        "title": "r/personalfinance - Reddit",
    },
    "reddit.com/submit": {
        "tree": REDDIT_SUBMIT,
        "title": "Create submission - Reddit",
    },
    "reddit.com/post_submitted": {
        "tree": REDDIT_CONFIRMATION,
        "title": "Post submitted - Reddit",
    },
    # Shopping pages
    "onestopmarket.com": {
        "tree": SHOPPING_HOME,
        "title": "OneStopMarket",
    },
    "onestopmarket.com/product": {
        "tree": SHOPPING_PRODUCT,
        "title": "HP Fax Machine - OneStopMarket",
    },
    "onestopmarket.com/wishlist": {
        "tree": SHOPPING_WISHLIST,
        "title": "My Wishlist - OneStopMarket",
    },
    # GitLab pages
    "gitlab.com/mr": {
        "tree": GITLAB_MR,
        "title": "Merge Request #1531 - GitLab",
    },
    # Classifieds
    "classifieds.com": {
        "tree": CLASSIFIEDS_HOME,
        "title": "Classifieds - Buy & Sell",
    },
}


# ---------------------------------------------------------------------------
# URL mapper (similar to AgentDAM's env_config.py URL_MAPPINGS)
# ---------------------------------------------------------------------------
URL_REVERSE_MAP = {
    "reddit.com": "reddit.com",
    "http://reddit.com": "reddit.com",
    "http://reddit.com/r/personalfinance": "reddit.com/r/personalfinance",
    "http://reddit.com/submit": "reddit.com/submit",
    "http://reddit.com/post_submitted": "reddit.com/post_submitted",
    "__REDDIT__": "reddit.com",
    "__REDDIT__/submit": "reddit.com/submit",
    "__REDDIT__/f/personalfinance": "reddit.com/r/personalfinance",
    "http://onestopmarket.com": "onestopmarket.com",
    "http://onestopmarket.com/product": "onestopmarket.com/product",
    "http://onestopmarket.com/wishlist": "onestopmarket.com/wishlist",
    "__SHOPPING__": "onestopmarket.com",
    "__SHOPPING__/": "onestopmarket.com",
    "__GITLAB__/a11yproject/a11yproject.com/-/merge_requests/1531": "gitlab.com/mr",
    "http://classifieds.com": "classifieds.com",
}


def resolve_url(url: str) -> Tuple[str, dict]:
    """Resolve a URL (possibly with __PLACEHOLDER__) to a page entry.

    Returns (canonical_key, page_dict) or ("unknown", default_page).
    """
    # Try direct match first
    if url in URL_REVERSE_MAP:
        key = URL_REVERSE_MAP[url]
        if key in PAGE_REGISTRY:
            return key, PAGE_REGISTRY[key]

    # Try matching without http prefix
    clean = url.replace("http://", "").replace("https://", "")
    if clean in URL_REVERSE_MAP:
        key = URL_REVERSE_MAP[clean]
        if key in PAGE_REGISTRY:
            return key, PAGE_REGISTRY[key]

    # Fuzzy match: check if any known key appears in the URL
    for known_key in PAGE_REGISTRY:
        if known_key in clean:
            return known_key, PAGE_REGISTRY[known_key]

    # Default: return a generic page
    return "unknown", {
        "tree": _generic_page(url),
        "title": url.split("/")[-1] or "Unknown Page",
    }


def _generic_page(url: str) -> str:
    return (
        f"Tab 0 (current): Web Page\n\n"
        f"[0] RootWebArea 'Web Page' focused: True\n"
        f"    [1] link 'Home'\n"
        f"    [2] link 'Back'\n"
        f"    [10] heading 'Page at {url[:60]}'\n"
        f"    [20] StaticText 'This page is a simplified placeholder.'\n"
    )


# ---------------------------------------------------------------------------
# Simulated environment that uses these pages
# ---------------------------------------------------------------------------
class RealisticMockBrowser:
    """A mock browser that returns realistic accessibility trees and responds
    to AgentDAM actions (click, type, goto, stop) with state changes.

    Usage:
        env = RealisticMockBrowser()
        obs = env.reset("__REDDIT__/f/personalfinance")
        # obs = {"text": <realistic accessibility tree>, "image": None}
        obs, reward, done, info = env.step(action)
    """

    def __init__(self, start_url: str = "reddit.com"):
        self.start_url = start_url
        self.current_key = "unknown"
        self.current_url = start_url
        self._typed_content: Dict[str, str] = {}  # element_id -> text
        self._submitted = False
        self._step = 0
        self._history: List[str] = []  # URL history

    def reset(self, *args, **kwargs) -> dict:
        self._step = 0
        self._typed_content = {}
        self._submitted = False
        self._history = [self.start_url]

        key, page = resolve_url(self.start_url)
        self.current_key = key
        self.current_url = self.start_url

        return {
            "text": page["tree"],
            "image": None,
        }

    def step(self, action: dict) -> Tuple[dict, float, bool, bool, dict]:
        """Execute a browser action and return (obs, reward, done, truncated, info)."""
        self._step += 1
        action_type = action.get("action_type", 0)
        info = {"page_url": self.current_url, "fail_error": ""}
        reward = 1.0
        done = False

        # --- GOTO_URL (13) ---
        if action_type == 13:
            new_url = action.get("url", self.current_url)
            key, page = resolve_url(new_url)
            self.current_key = key
            self.current_url = new_url
            self._history.append(new_url)
            return {"text": page["tree"], "image": None}, reward, False, False, info

        # --- CLICK (6) ---
        if action_type == 6:
            element_id = action.get("element_id", "")
            # "Create Post" button on reddit home → navigate to submit
            if element_id == "580" and "reddit" in self.current_key:
                key, page = resolve_url("reddit.com/submit")
                self.current_key = key
                self.current_url = "reddit.com/submit"
                return {"text": page["tree"], "image": None}, reward, False, False, info
            # "Submit" link on reddit → navigate to submit
            if element_id in ("62", "66") and "reddit" in self.current_key:
                key, page = resolve_url("reddit.com/submit")
                self.current_key = key
                self.current_url = "reddit.com/submit"
                return {"text": page["tree"], "image": None}, reward, False, False, info
            # "Create submission" button on submit page
            if element_id == "217" and self.current_key == "reddit.com/submit":
                title = self._typed_content.get("84", "")
                body = self._typed_content.get("93", "")
                key, page = resolve_url("reddit.com/post_submitted")
                self.current_key = key
                self.current_url = "reddit.com/post_submitted"
                self._submitted = True
                self._submitted_title = title
                self._submitted_body = body
                return {"text": page["tree"], "image": None}, reward, False, False, info
            # "Add to Wish List" on shopping product → wishlist
            if element_id in ("105", "114", "124", "1760") and "onestopmarket" in self.current_key:
                key, page = resolve_url("onestopmarket.com/wishlist")
                self.current_key = key
                self.current_url = "onestopmarket.com/wishlist"
                return {"text": page["tree"], "image": None}, reward, False, False, info
            # "Comment" button on GitLab
            if element_id == "92" and self.current_key == "gitlab.com/mr":
                comment = self._typed_content.get("91", "")
                self._submitted = True
                self._submitted_comment = comment
                # Stay on same page but show updated state
                page = PAGE_REGISTRY.get(self.current_key, {"tree": _generic_page(self.current_url)})
                updated_tree = page["tree"].replace(
                    'textbox \'Comment\' required: False focused: True',
                    'StaticText \'Comment submitted.\''
                )
                return {"text": updated_tree, "image": None}, reward, False, False, info
            # "Save Note" on wishlist
            if element_id == "522" and self.current_key == "onestopmarket.com/wishlist":
                note = self._typed_content.get("521", "")
                self._submitted = True
                self._submitted_note = note
                page = PAGE_REGISTRY.get(self.current_key, {"tree": _generic_page(self.current_url)})
                updated_tree = page["tree"].replace(
                    'StaticText \'Your wishlist is empty.\'',
                    'StaticText \'Note saved! Your wishlist has been updated.\''
                )
                return {"text": updated_tree, "image": None}, reward, False, False, info
            # Generic click: just stay on the same page
            page = PAGE_REGISTRY.get(self.current_key, {"tree": _generic_page(self.current_url)})
            return {"text": page["tree"], "image": None}, reward, False, False, info

        # --- TYPE (7) ---
        if action_type == 7:
            element_id = action.get("element_id", "")
            # Decode text from key IDs
            text_ids = action.get("text", [])
            if isinstance(text_ids, list) and text_ids:
                from browser_env.actions import _id2key
                chars = []
                for tid in text_ids:
                    if tid < len(_id2key):
                        k = _id2key[tid]
                        if k == "Enter":
                            chars.append("\n")
                        elif len(k) == 1:
                            chars.append(k)
                text = "".join(chars)
            else:
                text = str(text_ids)

            self._typed_content[element_id] = text
            page = PAGE_REGISTRY.get(self.current_key, {"tree": _generic_page(self.current_url)})
            return {"text": page["tree"], "image": None}, reward, False, False, info

        # --- STOP (17) ---
        if action_type == 17:
            done = True
            page = PAGE_REGISTRY.get(self.current_key, {"tree": _generic_page(self.current_url)})
            return {"text": page["tree"], "image": None}, reward, True, False, info

        # --- GO_BACK (11) ---
        if action_type == 11:
            if len(self._history) > 1:
                self._history.pop()
                prev_url = self._history[-1]
                key, page = resolve_url(prev_url)
                self.current_key = key
                self.current_url = prev_url
                return {"text": page["tree"], "image": None}, reward, False, False, info

        # --- SCROLL (1) ---
        if action_type == 1:
            page = PAGE_REGISTRY.get(self.current_key, {"tree": _generic_page(self.current_url)})
            return {"text": page["tree"], "image": None}, reward, False, False, info

        # Default: return current page
        page = PAGE_REGISTRY.get(self.current_key, {"tree": _generic_page(self.current_url)})
        return {"text": page["tree"], "image": None}, reward, False, False, info

    def close(self):
        pass

    @property
    def submitted_content(self) -> Optional[Dict[str, str]]:
        """Return what was submitted, if anything."""
        if not self._submitted:
            return None
        return {
            "title": getattr(self, "_submitted_title", ""),
            "body": getattr(self, "_submitted_body", ""),
            "comment": getattr(self, "_submitted_comment", ""),
            "note": getattr(self, "_submitted_note", ""),
        }
