;; -*- lexical-binding: t; -*-

(TeX-add-style-hook
 "cfd_solver_details"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-class-options
                     '(("article" "english")))
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("fontenc" "T1") ("inputenc" "latin9") ("geometry" "") ("amsmath" "") ("amssymb" "") ("caption" "") ("subcaption" "") ("float" "") ("cancel" "") ("empheq" "") ("tcolorbox" "") ("tikz" "") ("tkz-euclide" "") ("babel" "")))
   (TeX-run-style-hooks
    "latex2e"
    "article"
    "art10"
    "fontenc"
    "inputenc"
    "geometry"
    "amsmath"
    "amssymb"
    "caption"
    "subcaption"
    "float"
    "cancel"
    "empheq"
    "tcolorbox"
    "tikz"
    "tkz-euclide"
    "babel")
   (TeX-add-symbols
    '("widefbox" 1))
   (LaTeX-add-labels
    "fig:grid"
    "eq:ns_equation"
    "eq:cont_equation"
    "fig:grid_x"
    "fig:grid_y"
    "fig:contGrid"))
 :latex)

