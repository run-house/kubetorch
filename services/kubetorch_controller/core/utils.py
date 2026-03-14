import jinja2
import yaml


def _get_rendered_template(
    template_file: str, template_dir: str, **template_vars
) -> str:
    """Helper function to set up and render a template."""
    template_loader = jinja2.FileSystemLoader(searchpath=template_dir)
    template_env = jinja2.Environment(
        loader=template_loader,
        keep_trailing_newline=True,
        trim_blocks=True,
        lstrip_blocks=True,
        enable_async=False,
        autoescape=False,
    )
    template = template_env.get_template(template_file)
    return template.render(**template_vars)


def load_template(template_file: str, template_dir: str, **template_vars) -> dict:
    """Load and render a single YAML document template."""
    rendered = _get_rendered_template(template_file, template_dir, **template_vars)
    return yaml.safe_load(rendered)
