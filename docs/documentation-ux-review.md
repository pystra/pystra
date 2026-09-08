**PySTRA v2 documentation: UX assessment**

Reviewed 9 September 2026 against `v2.0` at `a66e7cd`.

The documentation has a strong scientific foundation and a much clearer hierarchy after the recent reorganisation. Its weakest areas are helping someone start, choose an appropriate method, and complete an engineering task. The next improvement should be a clearer route through the existing material, supported by a small number of practical guides.

This is a heuristic assessment. It covers the complete generated page inventory and navigation, all 19 maintained tutorial outlines, the entry and installation pages, guides, theory organisation, representative API pages, and contributor instructions. The inventory contains 282 HTML pages with article content after excluding source-code views; 222 are generated API pages. Page counts measure the shape of the documentation, not its quality. The in-app browser was unavailable: search ranking, keyboard operation, responsive layout, rendered equations, and colour contrast still require interactive verification. This is not a user study or an accessibility certification.

**The scientific content and recent improvements are worth preserving.** The five tutorial categories, seven API groups and ten theory pages provide a workable foundation. Executed examples, independent benchmark references, explicit convergence status, and distinctions between approximations, sampling uncertainty and surrogate diagnostics support informed use. The new plotting helpers reduce the amount of figure code readers must adapt. Source attribution and limitations should remain close to the results they qualify.

**Comparable projects suggest specific improvements.** These are observed design patterns and my recommendations for applying them, not claims that those sites have been proven more usable in a controlled comparison.

| Reference | Observed pattern | Application to PySTRA |
| --- | --- | --- |
| [NumPy documentation](https://numpy.org/doc/stable/) and [SciPy documentation](https://docs.scipy.org/doc/scipy/) | Landing pages explain the purpose of getting-started material, guides and API reference. | Give readers explicit destinations with a sentence describing what each helps them do. |
| [scikit-learn: choosing an estimator](https://scikit-learn.org/stable/machine_learning_map.html) | Provides an entry point specifically for choosing an algorithm. | Add a reliability-method selection guide based on problem properties, required assumptions, cost and validation needs. |
| [OpenTURNS common use cases](https://openturns.github.io/openturns/latest/usecases/usecases.html) | Defines reusable problems and connects each to examples using it. | Make the beam, four-branch system, truss, hat and Rosenblatt-ordering studies discoverable through a benchmark catalogue. Retain PySTRA's structural-reliability scope. |
| [Diátaxis](https://diataxis.fr/) | Distinguishes learning, completing a task, looking up a specification and understanding a concept. | Preserve tutorials, reference and theory; strengthen the currently thin task-oriented User guide. This does not require four rigidly separate websites. |
| [PyData theme version navigation](https://pydata-sphinx-theme.readthedocs.io/en/stable/user_guide/version-dropdown.html) | Supports labelled documentation versions and switching between corresponding pages. | Keep the existing theme and give stable and development documentation unmistakable identities. Publish the corresponding versions before adding switcher targets. |

**The main journeys reveal where readers still have to do too much work.**

| Reader's aim | Current route and friction | Better outcome |
| --- | --- | --- |
| Run a first reliability analysis | Installation offers the stable command first; Tutorials → Getting started → first analysis then introduces three correlated variables, FORM, two SORM approaches and simulation. | A version-matched install route and a short resistance-minus-load example with an independently known answer, followed by the richer existing tutorial. |
| Choose an analysis method | Method names are available in API, theory and examples, but there is no central selection guide. | A comparison explaining when FORM, SORM, simulation, system FORM or active learning is a useful starting point, and how to check the result. |
| Calibrate code factors | The tutorial and new API names are clear, but the User guide has no calibration entry. | A task guide linking model specification, candidate factors, normalized reliability, factor derivation and verification. |
| Find a known API object | One navigational route is API → calibration category → calibration package → normalized module → CodeCalibration. The category mentions the name as plain code text. | Direct object links from the category, plus the full module reference for detailed browsing. |
| Reproduce a paper example | Benchmarks are distributed among systems and active-learning tutorials. Downloading a notebook is not exposed as a direct article action. | A benchmark catalogue with model assumptions, source, reference method and links to runnable notebook bundles. |
| Resolve an unexpected result | Useful cautions exist within method pages; there is no central troubleshooting route. | Task pages for FORM nonconvergence, zero observed failures, transformation choices, sampling precision and surrogate termination. |

**Priority 0: fix version and content inconsistencies before polishing the layout.**

1. **Make the version visible and make the first installation command match it.** The generated homepage's `Version:` field contains a GitHub total-download-count badge labelled “version”, although its browser title correctly identifies `2.0.0.dev0`. [The homepage source](source/index.rst) overrides the normal version substitution with that badge; its legacy badge markup also leaves literal `..` text in the generated article. The [installation page](source/install.md) correctly distinguishes stable installation from a v2 checkout, but presents the stable command first. A reader can copy it and then follow incompatible v2 examples. Lead development docs with a visible “2.0 development” label and matching installation instructions, including a command to inspect the installed version. Keep stable instructions with an explicit stable-docs link. A version switcher complements this; it does not replace an unambiguous page label.

2. **Remove conflicting statements about implemented capabilities.** [Active-learning theory](source/theory/active_learning.rst) says importance-sampling integration remains separate work; the [guide](source/active_learning.rst) describes the implemented `ImportanceSamplingEstimator`. The [README](../README.rst) still claims there is no limitation on the limit-state function, despite the documented requirements of individual algorithms. Review such claims against the implemented capability descriptions. Keep release plans and implementation history in migration/release material, and give each current capability one maintained description.

3. **Unify contributor instructions.** The [developer page](source/developer.rst) recommends Python 3.12, manual dependency installation, indexing examples directly in the tutorial root and routine `make clean`. [CONTRIBUTING.md](../CONTRIBUTING.md) specifies Python 3.13 for docs/formatting, extras, nested tutorial discovery and cached builds. Routine cleaning defeats the new notebook cache. Make the developer page a readable rendering of one authoritative workflow and link separate extension recipes from it.

**Priority 1: improve starting, choosing and doing.**

4. **Rewrite the homepage around the work users came to do.** Its opening currently consists of badges, bibliographic metadata, general prose and a large contents list. Give it a concise scope statement and routes to “Run a reliability analysis”, “Calibrate code factors” and “Compare assessment scenarios”. Add a prominent first example and method-selection link. Put contributor and release information in secondary navigation. Keep the recently added groups in the sidebars; remove the need for the homepage to reproduce the entire contents tree.

5. **Make the User guide a practical manual.** Its current five entries are systems, Strong Maximum Test, copulas, active learning and plotting. Core modelling, ordinary FORM/SORM, simulation, calibration and assessment have no corresponding guide entries. These are missing routes, even where the underlying information already exists in tutorials and theory. Start with choosing a method, specifying a model and dependence, interpreting results and convergence, calibrating factors, comparing assessment scenarios, and plotting/reporting. An assessment guide should describe the currently supported scenario workflow and its assumptions, without implying an unimplemented deterioration or Bayesian-updating framework.

6. **Give beginners one small success before surveying methods.** Keep the existing first tutorial as an introduction to comparing methods. Precede it with a short independent-normal `R-S` example: define inputs, run FORM, inspect convergence and probability, check the analytic result. Introduce the model → analysis → result pattern there. Put distribution-adapter timing later in the learning path; it is not normally a beginner's next task. The normalized-calibration tutorial is a good model for a focused purpose, explicit assumptions and interpretation.

7. **Curate API entry points and add return links to explanations.** The grouping is useful, but it still exposes package layout as the next navigation layer. Put linked public classes/functions directly on category pages with one-line purposes. Show the recommended import, minimal use and result contract before extensive details. The [class template](source/_templates/custom-class-template.rst) includes inherited members; the generated `Form` page gives computational stages such as `compute_alpha` substantial prominence alongside `run`. Separate everyday operations from algorithm-extension details without silently removing supported API documentation. The inspected `Form` and `CodeCalibration` pages do not link directly to their worked tutorials. Add a small, consistent set of links to the relevant guide, example and theory page. Preserve acronym spellings.

8. **Make examples easy to select and take away.** Add concise descriptions to category links and label optional dependencies and substantial execution cost. Reading time and execution time are different; report timings only with a stated environment. In the current HTML, none of the 19 tutorial articles provides a direct notebook-download link; the theme's GitHub edit link is not a download action. Provide notebook downloads, bundling `literature_benchmarks.py` where required, and add copy buttons for code. The longest DDO notebook combines several distinct decisions: consider a core steel-bar decision tutorial and a separate target-table tutorial, with shared support code only where it helps reuse. Keep one canonical file per example; a second benchmark catalogue should cross-link those files rather than duplicate them.

**Priority 2: improve continuity, readability and discovery.**

9. **Finish the editorial work across guides, theory and reference.** The recent tutorial pass establishes a common base; it does not yet make the entire documentation consistent. Several tutorial section headings remain in title case while others use sentence case. The theory split retains book-like references to “preceding sections”, duplicate topic headings and local contents lists alongside the theme's page navigation. Normal coordinates use `Z` in older theory and `u` in newer examples. State the notation correspondence explicitly, keep source notation where reproducing a paper, and use a common notation guide. Definitions, assumptions and equations belong in theory; task sequence belongs in the guide; signatures, shapes and failure behaviour belong in reference. Retain concise operational limitations in each place where they affect a user's choice.

10. **Connect related pages in both directions.** In the inspected generated articles, fundamentals, simulation theory and systems theory have no direct links to their worked examples. The theory page about design points links to the Strong Maximum Test example but not the introductory FORM/SORM example. Use consistent “Worked example”, “API” and “Theory” links. Ensure someone arriving from a search engine can understand the page's role and continue without returning to the homepage.

11. **Treat accessibility and search as verification work.** The 19 rendered tutorials contain 21 plot images, all with filename-based alternative text. Add descriptive captions and meaningful alternatives; explain the conclusion in surrounding prose and avoid relying on colour alone. The existing dark-mode DataFrame styling is useful. Check figures, equations, wide tables, focus order, sidebar toggles and code copying at narrow widths and 200% zoom in an actual browser. Search is already present; do not replace it speculatively. Test representative queries such as “FORM”, “code calibration”, “Nataf” and “zero failures” before deciding whether titles, indexing or ranking need changes. Preserve existing URLs and anchors through future moves, particularly across the 1.x/v2 transition.

**A compact navigation model would keep the current structure and change its emphasis.**

| Main destination | Purpose and contents |
| --- | --- |
| Get started | Matching installation; first analysis; basic concepts; suggested next steps. |
| User guide | Method selection; modelling; results and troubleshooting; systems/dependence; calibration and assessment; plotting. |
| Examples | The existing five categories, guided learning paths, and a cross-linked benchmark catalogue. |
| API reference | The existing seven groups, with direct public-object links and deeper module detail. |
| Theory | The existing ten topic pages, with common notation and links to examples. |

Keep search and the version identity persistently available. Put migration, release notes, references/citation guidance, contributing and project information in a secondary menu or footer. The rendered header currently shows Installation, Migration, Tutorials, User guide and API before putting Theory and other sections into its overflow menu. Moving Migration into the secondary group frees a primary position for Theory without adding another top-level choice. Existing users should still have a conspicuous migration link on the development landing page.

**I would implement this in three tranches.** First, fix the version/install experience and conflicting instructions, then create the first-analysis route. Second, add the method-selection and task guides, direct API links and cross-links between existing pages. Third, improve example downloads/discovery, consolidate long tutorials where justified, and complete browser-based accessibility and search checks. None of this requires changing the theme, replacing Sphinx or expanding the library into general UQ.

**Success should be judged with concrete tasks.** A new user should install the matching version and reproduce the first result without opening migration documentation. From the User guide, an engineer should find a suitable starting method and its limitations within two content pages. From an API category, a named public object should be directly reachable. A downloaded benchmark should run with its supplied helpers and documented dependencies. A reader should be able to navigate between a method's guide, example, API and theory in either direction. Finally, test keyboard navigation, search, narrow-screen reading and zoom with representative users; build success and passing numerical tests do not establish those UX outcomes.


**Implementation on v2.0**

The baseline assessment above is retained as the rationale for the changes.
The implementation adds five primary navigation destinations, a persistent
version label, a version-matched installation path, a checked first analysis,
practical task guides, direct API links and a benchmark catalogue. The contributor
page renders the authoritative `CONTRIBUTING.md`, with distribution-extension
material in a separate recipe. The steel-bar decision and target-table studies
are now separate, independently runnable examples.

All 21 maintained notebooks have dependency declarations, notebook and bundle
downloads, and guide/API/theory links. All 21 plotted outputs have descriptive
alternatives and visible captions. Calibration envelopes use hatching; PCE
selection and Strong Maximum Test groups use distinct markers. The existing
page URLs and meaningful section anchors are retained; removed local contents
widgets and generated pandas table identifiers are not treated as public anchors.

`python scripts/check_docs.py docs/build/html` checks local files and anchors,
notebook/bundle freshness, helper contents, figure alternatives and entry-page
markup. CI also executes every indexed notebook and runs the guide/API doctests.
The local validation includes an extracted hat-benchmark bundle, confirming
that its helper is imported from the downloaded directory.

Interactive verification remains outstanding because the in-app browser was
unavailable. Static checks and inspection of the generated plot images do not
establish keyboard usability, responsive layout, zoom behaviour, colour contrast,
or search-result ranking. Those checks remain a release-review task; the theme's
existing search and accessibility controls have been retained.
