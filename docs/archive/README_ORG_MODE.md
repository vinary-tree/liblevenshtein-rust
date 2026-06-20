# Org-Mode Documentation

## Overview

The file `RHOLANG_STRUCTURAL_CORRECTION.org` is an Emacs Org-mode version of the design document that can be compiled to PDF with proper LaTeX formatting.

## Files

- **RHOLANG_STRUCTURAL_CORRECTION.md** - Original GitHub-flavored Markdown (7,758 lines)
- **RHOLANG_STRUCTURAL_CORRECTION.org** - Org-mode version with LaTeX (7,593 lines)
- **convert_to_org_advanced.py** - Converter script (Python 3)

## Why Org-Mode?

1. **LaTeX Math** - Proper mathematical typesetting with equations
2. **PDF Export** - Professional PDF output via LaTeX
3. **Better Structure** - Org-mode's hierarchical structure
4. **GitHub Support** - GitHub renders .org files natively
5. **Emacs Integration** - Full Emacs editing features

## Compiling to PDF

### Method 1: Emacs (Interactive)

```bash
emacs docs/RHOLANG_STRUCTURAL_CORRECTION.org
```

Then in Emacs:
- Press `C-c C-e` (export menu)
- Press `l` (LaTeX export)
- Press `p` (export to PDF and open)

### Method 2: Command Line (Batch)

```bash
cd docs
emacs --batch \
  --eval "(require 'ox-latex)" \
  --visit=RHOLANG_STRUCTURAL_CORRECTION.org \
  --funcall org-latex-export-to-pdf
```

This will create `RHOLANG_STRUCTURAL_CORRECTION.pdf` in the same directory.

### Method 3: Using org-mode script

```bash
emacs --batch \
  --eval "(progn
    (require 'ox-latex)
    (find-file \"docs/RHOLANG_STRUCTURAL_CORRECTION.org\")
    (org-latex-export-to-pdf))"
```

## Requirements

### For PDF Generation

You need a LaTeX distribution installed:

**Linux:**
```bash
# Debian/Ubuntu
sudo apt-get install texlive-full emacs

# Arch Linux
sudo pacman -S texlive-most emacs

# Fedora
sudo dnf install texlive-scheme-full emacs
```

**macOS:**
```bash
brew install --cask mactex
brew install emacs
```

**Windows:**
- Install [MiKTeX](https://miktex.org/download)
- Install [Emacs for Windows](https://www.gnu.org/software/emacs/download.html)

### LaTeX Packages Used

The org file includes these LaTeX packages:
- `amsmath` - Mathematical typesetting
- `amssymb` - Mathematical symbols
- `amsthm` - Theorem environments
- `mathtools` - Enhanced math typesetting
- `unicode-math` - Unicode math symbols
- `listings` - Code listings
- `xcolor` - Color support
- `hyperref` - PDF hyperlinks
- `geometry` - Page layout
- `fancyvrb` - Enhanced verbatim
- `tikz` - Diagrams (if needed)

## Features

### Mathematical Equations

Inline math: $T_{total} = T_{lexical} \otimes T_{syntactic}$

Display math:
```latex
\[
  T_{total} = T_{semantic} \circ T_{structural} \circ T_{syntactic} \circ T_{lexical}
\]
```

### Code Blocks

```rust
#+begin_src rust
pub struct SyntaxAwareCorrector {
    language: Language,
    transition_map: HashMap<(String, usize), Vec<String>>,
}
#+end_src
```

### Theorem Environments

```latex
\begin{theorem}
WFSTs are closed under composition.
\end{theorem}
```

## Viewing Without Compilation

### GitHub

GitHub automatically renders .org files:
```
https://github.com/your-repo/docs/RHOLANG_STRUCTURAL_CORRECTION.org
```

### Emacs (Plain View)

```bash
emacs docs/RHOLANG_STRUCTURAL_CORRECTION.org
```

Use standard Emacs org-mode navigation:
- `TAB` - Cycle visibility of current heading
- `S-TAB` - Cycle visibility globally
- `C-c C-n` / `C-c C-p` - Next/previous heading

## Regenerating from Markdown

If you update the .md file:

```bash
python3 docs/convert_to_org_advanced.py
```

This will regenerate the .org file from the latest markdown.

## PDF Output Customization

Edit the org file header to customize PDF output:

```org
#+LATEX_CLASS_OPTIONS: [11pt,a4paper]  # Change to [12pt,letterpaper] for US letter
#+LATEX_HEADER: \geometry{left=1in,right=1in,top=1in,bottom=1in}  # Adjust margins
```

## Troubleshooting

### "LaTeX Error: File not found"

Install missing LaTeX packages:
```bash
# Ubuntu/Debian
sudo apt-get install texlive-latex-extra texlive-fonts-extra

# Or install on-the-fly (MiKTeX)
# MiKTeX will prompt to install missing packages automatically
```

### "org-latex-export-to-pdf: Symbol's function definition is void"

Make sure Emacs has org-mode and LaTeX export loaded:
```elisp
(require 'ox-latex)
```

### Slow PDF generation

The document is large (7,500+ lines). PDF generation may take 1-2 minutes.
Be patient or compile in parts by editing the org file.

## Advantages Over Markdown

| Feature | Markdown | Org-Mode |
|---------|----------|----------|
| Math equations | Limited (inline code) | Full LaTeX support |
| PDF export | Requires pandoc | Native via LaTeX |
| Code blocks | Syntax highlighting | Executable + highlighting |
| Cross-references | Manual | Automatic |
| Bibliography | Not native | BibTeX integration |
| Theorems/Proofs | None | LaTeX environments |
| Page breaks | Not supported | Full control |
| Professional typography | No | Yes (via LaTeX) |

## License

Same as parent project.

## Questions?

For issues with:
- **Conversion script**: Check `convert_to_org_advanced.py`
- **Org-mode**: See https://orgmode.org/manual/
- **LaTeX errors**: Check TeXLive/MiKTeX logs
