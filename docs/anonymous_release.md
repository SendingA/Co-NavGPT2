# Anonymous source distribution

Use a source snapshot without `.git` for anonymous review. Existing commits,
remote URLs, branches and tags can identify contributors even after the current
source has been anonymized. A regular commit does not remove that history.

The source snapshot must exclude local credentials, editor settings, caches,
task records and experiment outputs. Do not distribute the working directory
as a whole. Keep third-party copyright and license notices intact.

Dataset and embodied assets are distributed separately through the resource
links in the [README](../README.md). Review their file metadata, sharing owner,
and the project website separately before submitting an anonymous artifact.
Source cleanup does not establish the anonymity of external resources.

## Portable configuration

Activate the `firenav` environment before running scripts. Shell experiment
launchers use `python` from the active environment, or the executable selected
by `FIRENAV_PYTHON`.

The project-specific YAML settings live under `firenav`. Docker uses the
`firenav` and `firenav-test` services, the `firenav:latest` image, and the
`FIRENAV_*` variables in [docker/.env.example](../docker/.env.example).
Update local custom configs and automation to these names when adopting this
source snapshot.

The optional ROS launchers require deployment endpoints supplied locally.
For a single robot, set `FIRENAV_ROBOT_ENDPOINT`. For multiple robots, set
`FIRENAV_ROBOT_0_ENDPOINT`, `FIRENAV_ROBOT_1_ENDPOINT`, and so on. Values must
be ZeroMQ endpoints such as `tcp://127.0.0.1:5557`; replace the loopback address
with the endpoint for your own deployment. No lab network addresses are
embedded in the source.

Generated experiment manifests and logs can contain resolved filesystem paths
and runtime configuration. Review them separately before sharing; the source
snapshot excludes them.
