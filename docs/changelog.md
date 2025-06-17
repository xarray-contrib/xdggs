# Changelog

## 0.2.1 (_unreleased_)

### New features

### Bug fixes

### Documentation

- Documentation Contributer Guide + Github Button ({pull}`137`)

### Internal changes

## 0.2.0 (2025-02-12)

## New features

- allow adding additional coords to the cell inspection table in the map ({pull}`122`)
- allow passing `matplotlib` colormap objects to `explore` ({pull}`120`)
- support plotting multi-dimensional data ({pull}`124`)
- allow overriding the grid info data using the function or a new accessor ({pull}`63`, {pull}`121`)

## Bug fixes

- use explicit `arrow` API to extract cell coordinates ({issue}`113`, {pull}`114`)
- correct the `HealpixIndex` `repr` ({pull}`119`)

## Internal changes

- add initial set of tests for `explore` ({pull}`127`)
- adapt to recent changes on RTD ({pull}`122`)

## 0.1.1 (2024-11-25)

### Bug fixes

- properly reference the readme in the package metadata ({pull}`106`)

## 0.1.0 (2024-11-25)

### Enhancements

- derive cell boundaries ({pull}`30`)
- add grid objects ({pull}`39`, {pull}`57`)
- decoder function ({pull}`47`, {pull}`48`)
- rename the primary grid parameter to `level` ({pull}`65`)
- interactive plotting with `lonboard` ({pull}`67`)
- expose example datasets through `xdggs.tutorial` ({pull}`84`)
- add a preliminary logo ({pull}`101`, {pull}`103`)

### Bug fixes

- fix the cell centers computation ({pull}`61`)
- work around blocked HTTP requests from RTD to github ({pull}`93`)

### Documentation

- create a readme ({pull}`70`)
- create the documentation ({pull}`79`, {pull}`80`, {pull}`81`, {pull}`89`)
- fix headings in tutorials ({pull}`90`)
- rewrite the readme ({pull}`97`)

### Internal changes

- replace `h3` with `h3ronpy` ({pull}`28`)
- setup CI ({pull}`31`)
- tests for the healpix index ({pull}`36`)
- testing utils for exception groups ({pull}`55`)

## 0.0.1 (2023-11-28)

Preliminary version of `xdggs` python package.
