# Changelog

## [0.2.0](https://github.com/christianhbye/eigsep_mock_analysis/compare/eigsim-v0.1.0...eigsim-v0.2.0) (2026-04-06)


### Features

* always use float64 precision, bump croissant version ([f19ddcb](https://github.com/christianhbye/eigsep_mock_analysis/commit/f19ddcb652abbf8517fe949fc56fa1df31808b92))
* **eigsim:** add canonical simulation script and config-driven sky/orientations ([593d7ae](https://github.com/christianhbye/eigsep_mock_analysis/commit/593d7ae2453ce74996b54f5181e22647819427de))
* **eigsim:** add drive rotations, simulation wrapper, and YAML config ([0b1c645](https://github.com/christianhbye/eigsep_mock_analysis/commit/0b1c645cf6f173fa761ec5b573806d01a97d2f36))
* **eigsim:** add HEALPix-to-MWSS conversion script and data loaders ([83363bc](https://github.com/christianhbye/eigsep_mock_analysis/commit/83363bcccfc0163f299a60eab00ba6fe1479239d))
* **eigsim:** add receiver temperature and radiometer noise module ([6d06982](https://github.com/christianhbye/eigsep_mock_analysis/commit/6d06982c209b38a6c9834ac8d5dd12fe9ade55f5))
* **eigsim:** batch canonical sim with checkpoint/resume ([b9151b2](https://github.com/christianhbye/eigsep_mock_analysis/commit/b9151b209ddad31466023f3c043c7e2d45d81fb4))


### Bug Fixes

* **eigsim:** use nearest-neighbor interpolation for horizon HP-&gt;MWSS ([d67feb1](https://github.com/christianhbye/eigsep_mock_analysis/commit/d67feb1870afcc6dcdaf165ebae69e07d0c82956))


### Performance Improvements

* speed up simulator with several precomupte calls and combined rotations ([386c761](https://github.com/christianhbye/eigsep_mock_analysis/commit/386c76135b373d5d93ff9a1d3c24be47c6365b31))


### Documentation

* **eigsim:** add CLAUDE.md with project context for Claude Code ([31cddb1](https://github.com/christianhbye/eigsep_mock_analysis/commit/31cddb1f9ce76d26184abd331ec2b3ee39288026))
