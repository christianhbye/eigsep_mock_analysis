# Changelog

## [0.2.0](https://github.com/christianhbye/eigsep_mock_analysis/compare/eigsim-v0.1.0...eigsim-v0.2.0) (2026-09-15)


### ⚠ BREAKING CHANGES

* **eigsim:** load_config() and load_beam() now default to the v001 beam on the 52 HFSS channels (46.875-246.09 MHz, 3.906 MHz apart) instead of v000 on 50-250 MHz at 1 MHz. Pass "eigsep_v000" for the old behaviour.

### Features

* always use float64 precision, bump croissant version ([f19ddcb](https://github.com/christianhbye/eigsep_mock_analysis/commit/f19ddcb652abbf8517fe949fc56fa1df31808b92))
* **eigsim:** add canonical simulation script and config-driven sky/orientations ([593d7ae](https://github.com/christianhbye/eigsep_mock_analysis/commit/593d7ae2453ce74996b54f5181e22647819427de))
* **eigsim:** add compute_fgnd and correct_ground_loss ([27c3b2e](https://github.com/christianhbye/eigsep_mock_analysis/commit/27c3b2ee1f05635d4295619101d20ec241dbc60f))
* **eigsim:** add drive rotations, simulation wrapper, and YAML config ([0b1c645](https://github.com/christianhbye/eigsep_mock_analysis/commit/0b1c645cf6f173fa761ec5b573806d01a97d2f36))
* **eigsim:** add HEALPix-to-MWSS conversion script and data loaders ([83363bc](https://github.com/christianhbye/eigsep_mock_analysis/commit/83363bcccfc0163f299a60eab00ba6fe1479239d))
* **eigsim:** add receiver temperature and radiometer noise module ([6d06982](https://github.com/christianhbye/eigsep_mock_analysis/commit/6d06982c209b38a6c9834ac8d5dd12fe9ade55f5))
* **eigsim:** add simulate_path for one orientation per time sample ([#13](https://github.com/christianhbye/eigsep_mock_analysis/issues/13)) ([7b9e582](https://github.com/christianhbye/eigsep_mock_analysis/commit/7b9e582718aaa013909be7bf02549e887429f57c))
* **eigsim:** batch canonical sim with checkpoint/resume ([b9151b2](https://github.com/christianhbye/eigsep_mock_analysis/commit/b9151b209ddad31466023f3c043c7e2d45d81fb4))
* **eigsim:** default to the v001 HFSS bowtie beam ([ad1459a](https://github.com/christianhbye/eigsep_mock_analysis/commit/ad1459a6ec89c28bdb13919e65429bba25b061d4))
* horizon and levelling sensitivity for the D5 forward model (phase 1) ([#18](https://github.com/christianhbye/eigsep_mock_analysis/issues/18)) ([66946b8](https://github.com/christianhbye/eigsep_mock_analysis/commit/66946b8ac9c5ef54e9618d912065c3fcdadff0b8))


### Bug Fixes

* **eigsim:** convert horizon to boolean mask in canonical sim script ([3d456a9](https://github.com/christianhbye/eigsep_mock_analysis/commit/3d456a9d52aafe46f709840a149fb883f5bd6190))
* **eigsim:** import test_data helpers through the package ([1a36b4f](https://github.com/christianhbye/eigsep_mock_analysis/commit/1a36b4f0c7e881a8d8734f1167374e1b89d7fb6b))
* **eigsim:** precompute sky alm in croissant's frame of date ([754627c](https://github.com/christianhbye/eigsep_mock_analysis/commit/754627cd99efbed7b4fd437bce1a1b66865d2234))
* **eigsim:** use nearest-neighbor interpolation for horizon HP-&gt;MWSS ([d67feb1](https://github.com/christianhbye/eigsep_mock_analysis/commit/d67feb1870afcc6dcdaf165ebae69e07d0c82956))


### Performance Improvements

* speed up simulator with several precomupte calls and combined rotations ([386c761](https://github.com/christianhbye/eigsep_mock_analysis/commit/386c76135b373d5d93ff9a1d3c24be47c6365b31))


### Dependencies

* bump croissant-sim to v5.3.0.dev3 ([b2824f5](https://github.com/christianhbye/eigsep_mock_analysis/commit/b2824f5f01a79822cc04e93dd96349f92d87204e))
* pin croissant-sim to the v5.3.0.dev2 git tag ([1384c8b](https://github.com/christianhbye/eigsep_mock_analysis/commit/1384c8b8d286b02fc812bbfe34b266a8703d01d1))


### Documentation

* **eigsim:** add CLAUDE.md with project context for Claude Code ([31cddb1](https://github.com/christianhbye/eigsep_mock_analysis/commit/31cddb1f9ce76d26184abd331ec2b3ee39288026))
* repoint stale eigsep_cal interface.md references; keep path-mode plan as history ([d01f304](https://github.com/christianhbye/eigsep_mock_analysis/commit/d01f304b15325eea951be4c841316bbafd161c6d))
