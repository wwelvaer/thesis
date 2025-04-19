sudo apt-get update
sudo apt-get install texlive texlive-xetex texlive-luatex texlive-science texlive-bibtex-extra texlive-fonts-recommended texlive-fonts-extra biber chktex
lualatex main.tex
biber    main
lualatex main.tex
lualatex main.tex
