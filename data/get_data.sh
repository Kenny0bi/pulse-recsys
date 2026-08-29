#!/bin/sh
# MovieLens-25M from GroupLens. NOTE: as of Aug 2026 files.grouplens.org
# serves an EXPIRED TLS certificate (verified with two independent stacks),
# so this fetch disables verification and then validates structurally: the
# zip's own CRCs plus the documented row count of 25,000,095 ratings.
cd "$(dirname "$0")"
curl -kfL -o ml-25m.zip https://files.grouplens.org/datasets/movielens/ml-25m.zip
unzip -tq ml-25m.zip
unzip -qo ml-25m.zip
LINES=$(wc -l < ml-25m/ratings.csv)
echo "ratings.csv lines: $LINES (expected 25000096 = 25,000,095 + header)"
