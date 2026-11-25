# ComfyUI Registry Publishing Setup

This package includes the current Registry metadata structure in `pyproject.toml`.
Before publishing, replace these two explicit placeholders with the real values:

- `project.urls.Repository`: the canonical public GitHub repository URL
- `tool.comfy.PublisherId`: the publisher ID created/selected in the ComfyUI Registry

The Registry requires a package name, three-part semantic version, repository URL,
and publisher ID. The current package name is `kiri-image-loader`; check Registry
availability before first publication because node-pack names are globally unique.

The `.comfyignore` file excludes the local regression-test directory and Python cache
artifacts from the Registry archive. `requirements.txt` and `project.dependencies`
carry the optional `send2trash` dependency. The conventional
`NODE_CLASS_MAPPINGS` entry point does not require a custom `node_list.json`.

Official requirements: https://docs.comfy.org/registry/specifications
