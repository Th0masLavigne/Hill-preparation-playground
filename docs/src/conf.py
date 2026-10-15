# conf.py — Sphinx configuration for hill_playground.
from importlib.metadata import version as get_version
from pathlib import Path
import sys 

current_dir = Path(__file__).parent.resolve()
project_root = current_dir.parent.parent
sys.path.insert(0, str(project_root))
# Demo scripts import their sibling helpers by bare name (run from their own folder): viewcode needs them.
for demo_dir in sorted((project_root / "demos").glob("*/")):
	sys.path.insert(0, str(demo_dir))
# -----------------------------------------------------------------------------
# Project information
# -----------------------------------------------------------------------------
project = "hill_playground"
author = "Thomas Lavigne"
copyright = "2026, Thomas Lavigne"

try:
	release = get_version(project)
except Exception:
	release = "0.0.0"

# -----------------------------------------------------------------------------
# General configuration
# -----------------------------------------------------------------------------
extensions = [
	"sphinx.ext.napoleon",  # Google/NumPy style docstrings
	"sphinx_autodoc_typehints",  # Auto-links type hints
	"myst_parser",  # Markdown support
	"sphinx_copybutton",  # Copy button for code blocks
	"sphinx_design",  # Fancy UI elements
	"autoapi.extension",  # AutoAPI for automatic API docs
	"sphinx.ext.viewcode", # add source button
	"sphinx.ext.mathjax", # render latex
	# "sphinx.ext.autodoc",
]

# MyST (Markdown) configuration
source_suffix = {
	".rst": "restructuredtext",
	".md": "markdown",
}
myst_enable_extensions = [
	"amsmath",
	"dollarmath",
	"colon_fence",
	"deflist",
	"fieldlist",
    "substitution",
	"html_admonition",
]
myst_heading_anchors = 3
myst_dmath_allow_labels = True
myst_dmath_double_inline = True  # Allows $$ inline
myst_admonition_enable = True

# -----------------------------------------------------------------------------
# Autodoc Configuration
# -----------------------------------------------------------------------------
autodoc_default_options = {
	"members": True,
	"undoc-members": True,
	"private-members": False,
	"show-inheritance": True,
	"exclude-members": "__init__,__dataclass_fields__",
}
autodoc_typehints = "signature"
python_use_unqualified_type_names = True

# -----------------------------------------------------------------------------
# Napoleon Configuration 
# -----------------------------------------------------------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = False
napoleon_preprocess_types = True

napoleon_use_ivar = True
napoleon_use_param = False
napoleon_use_rtype = True

# -----------------------------------------------------------------------------
# AutoAPI Configuration 
# -----------------------------------------------------------------------------
autoapi_type = "python"
autoapi_dirs = [
	str(project_root / "src/hill_playground"),
	str(project_root / "tests"),
	str(project_root / "demos")
]
autoapi_ignore = [
	"*tests/integration/analytical_consolidation_1D/debug_ressources/*",
	"*conf.py*", "*setup.py*", "*/docs/*", "*/.pytest_cache/*",
	"*/__pycache__/*", "*/.ruff_cache/*",
	"tests/test_env.py",
	"*conftest.py",
]
autoapi_root = "api"
autoapi_add_toctree_entry = False
autoapi_keep_files = False # set True if debugging required remember to delete docs/src/api/
autoapi_generate_api_docs = True
autoapi_member_order = "bysource"
autoapi_python_use_implicit_namespaces = True
autoapi_template_dir = None
autoapi_own_page_level = "module"
autoapi_prepare_getattr = True

autoapi_options = [
	"members",
	# "undoc-members",
	"show-inheritance",
	"imported-members",  
]

# -----------------------------------------------------------------------------
# HTML Output
# -----------------------------------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_title = project
html_static_path = ["_static"]
html_theme_options = {
	"navigation_with_keys": True,
	"titles_only": False,
	"includehidden": True,
	"navigation_depth": 4,
	"sticky_navigation": True,
	"collapse_navigation": False,
}

copybutton_prompt_text = r">>> |\.\.\. "
copybutton_prompt_is_regexp = True

extensions.append("sphinx.ext.intersphinx")
intersphinx_mapping = {
	"python": ("https://docs.python.org/3", None),
	"numpy": ("https://numpy.org/doc/stable", None),
}

# -----------------------------------------------------------------------------
# Custom Scripts
# -----------------------------------------------------------------------------
def autodoc_skip_member(app, what, name, obj, skip, options):
	if what == "module":
		try:
			if hasattr(obj, "__qualname__") and "." in obj.__qualname__:
				return True
		except Exception:
			pass
	return skip

def generate_simple_toc(app):
	src_dir = Path(app.srcdir)
	api_root = src_dir / "api"
	toc_file = src_dir / "api_toc.md"
	
	if not api_root.exists():
		return

	pillars = [
		("hill_playground", "Core Package"),
		("demos", "Simulation Demos"),
		("tests/integration", "Integration Tests"),
		("tests/unit", "Unit Tests"),
		("tests/environment", "Environment Tests"),
	]

	def ensure_index_exists(folder_path, title, current_rel_path):
		"""
		Recursively creates index.rst files and includes READMEs if found in source.
		"""
		idx_path = folder_path / "index.rst"
		seen_names = set()
		children = []

		# Pass 1: Directories
		for item in folder_path.iterdir():
			if item.is_dir() and item.name != "index.rst":
				folder_label = item.name.replace('_', ' ').title()
				# Pass the updated relative path to the next level
				ensure_index_exists(item, folder_label, f"{current_rel_path}/{item.name}")
				children.append(f"{folder_label} <{item.name}/index>")
				seen_names.add(item.name)

		# Pass 2: Files
		for item in folder_path.iterdir():
			if item.suffix == ".rst" and item.name != "index.rst":
				raw_stem = item.stem
				leaf_name = raw_stem.split('.')[-1]
				
				# Prevent directory/file collision
				if leaf_name in seen_names:
					continue
				
				clean_label = leaf_name.replace('_', ' ').title()
				children.append(f"{clean_label} <{raw_stem}>")
		
		# Write the index file
		if children:
			with idx_path.open("w", encoding="utf-8") as f:
				f.write(f"{title}\n" + "=" * len(title) + "\n\n")

				actual_source_dir = project_root / current_rel_path
				readme_file = actual_source_dir / "README.md"
				
				if readme_file.exists():
					depth = len(current_rel_path.split('/')) + 3
					ups = "../" * depth

					f.write(".. raw:: html\n\n   <hr>\n\n")
					# Include the README using the calculated relative path
					f.write(f".. include:: {ups}{current_rel_path}/README.md\n")
					f.write("   :parser: myst_parser.sphinx_\n\n")
					f.write(".. raw:: html\n\n   <hr>\n\n")
				
				f.write(".. contents::\n   :local:\n\n")
				f.write(".. toctree::\n   :maxdepth: 2\n   :titlesonly:\n\n")
				for child in sorted(set(children)):
					f.write(f"   {child}\n")

	valid_sections = []
	for rel_path, title in pillars:
		full_path = api_root / rel_path
		if full_path.exists():
			# Standardize the path for the Core Package which lives in 'src/'
			path_in_source = f"src/{rel_path}" if rel_path == "hill_playground" else rel_path
			ensure_index_exists(full_path, title, path_in_source)
			valid_sections.append(f"{title} <api/{rel_path}/index>")

	with toc_file.open("w", encoding="utf-8") as f:
		f.write("# API Documentation\n\n")
		f.write("```{toctree}\n:maxdepth: 6\n:titlesonly:\n\n")
		for section in valid_sections:
			f.write(f"{section}\n")
		f.write("```\n")


def skip_globals(app, what, name, obj, skip, options):
	"""
	Skipping module-level variables (global data) prevents things like 
	'comm', 'rank', and 'parent_dir' from appearing in the docs.
	"""
	# 'data' refers to top-level variables in a module
	if what == "data":
		return True
	
	return skip

def setup(app):
	app.connect("builder-inited", generate_simple_toc)
	app.connect("autodoc-skip-member", autodoc_skip_member)
	
	app.connect("autoapi-skip-member", skip_globals)

	if "custom.css" not in app.registry.css_files:
		app.add_css_file("custom.css")