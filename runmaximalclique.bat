@echo off

set MAX_CLIQUE=%1$

bin\mace.exe MqVe -l %MAX_CLIQUE% features\match.grh features\match_maximal_clique.grh
