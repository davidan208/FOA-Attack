#!/usr/bin/env bash
# Progress of the FGSM ablation:  watch -n 1 bash monitor.sh
cd "$(dirname "$0")"
R=${ROOT:-ablation_base_adaptive}
count() { find "$1" -path '*/_smoke' -prune -o -name '*.png' -print 2>/dev/null | wc -l; }

total=0
for d in "$R"/A/fgsm "$R"/B/fgsm "$R"/AB/fgsm; do
    n=$(count "$d"); total=$((total + n))
    printf "%-36s %3s/100\n" "$d" "$n"
done
echo "TOTAL: $total/300"
echo
echo "Last saved : $(grep 'Saved:' "$R/runner.log" 2>/dev/null | tail -1)"
echo "Now        : $(tail -c 400 "$R/runner.log" 2>/dev/null | tr '\r' '\n' | grep -v '^\s*$' | tail -1)"
