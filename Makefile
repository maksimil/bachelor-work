OUTDIR=./out
OBJDIR=./obj
OVERLEAFDIR=./overleaf

BASENAME=template
OUTNAME=work1

BASETEX=./$(BASENAME).tex
BASEOBJ=$(OBJDIR)/$(BASENAME).pdf
BASEOUT=$(OUTDIR)/$(OUTNAME).pdf
GSOUT=$(OUTDIR)/$(OUTNAME)-gs.pdf

LOGSECTION=@echo "\033[32m==== $(1) =====\033[0m"

SET_PREVIEWER=-e '$$pdf_previewer=q[zathura %S];'
SET_XELATEX=-e '$$pdflatex=q[xelatex %O %S];'
# SET_PATCHES=-e 'ensure_path(q[TEXINPUTS], q[./patches]);'

LATEX=latexmk -pdf -outdir=$(OBJDIR) $(SET_PREVIEWER) $(SET_XELATEX)

.PHONY: watch build filesys clean

filesys:
	$(call LOGSECTION, Creating filesystem)
	[ ! -d $(OUTDIR) ] && mkdir -p $(OUTDIR) || true
	[ ! -d $(OBJDIR) ] && mkdir -p $(OBJDIR) || true

clean:
	rm -rf $(OUTDIR) $(OBJDIR)

watch: filesys
	$(call LOGSECTION, Starting watch)
	$(LATEX) -pvc $(BASETEX)

build: filesys
	$(call LOGSECTION, Starting build)
	$(LATEX) $(BASETEX)
	$(call LOGSECTION, Optimizing)
	pdfsizeopt --do-require-image-optimizers=no --quiet $(BASEOBJ) $(BASEOUT)

gs: filesys build
	gs \
       -o $(GSOUT) \
       -sDEVICE=pdfwrite \
       -sColorConversionStrategy=Gray \
       -dProcessColorModel=/DeviceGray \
	   $(BASEOUT)

overleaf-package:
	rm -rf $(OVERLEAFDIR)

	mkdir $(OVERLEAFDIR)
	cp -r main.tex math.tex refs.bib template.tex $(OVERLEAFDIR)

	mkdir $(OVERLEAFDIR)/plots
	cp -r plots/pdf $(OVERLEAFDIR)/plots
