# MCDM Summary
Criteria: minimize time & memory, maximize accuracy.
Weights: time=0.5, memory=0.2, accuracy=0.3

## Weighted Sum Ranking
1. neural_sdf — score=0.9981 (time=0.0876, memory=209996, accuracy=1.000)
2. trimesh — score=0.9922 (time=0.2930, memory=2212297, accuracy=1.000)
3. aabb — score=0.6990 (time=0.0113, memory=2352302, accuracy=0.833)
4. kdtree — score=0.6962 (time=0.0713, memory=4998065, accuracy=0.833)
5. point_cloud — score=0.3000 (time=20.6357, memory=411424400, accuracy=1.000)

## TOPSIS Ranking
1. neural_sdf — closeness=0.9966 (time=0.0876, memory=209996, accuracy=1.000)
2. trimesh — closeness=0.9872 (time=0.2930, memory=2212297, accuracy=1.000)
3. aabb — closeness=0.9575 (time=0.0113, memory=2352302, accuracy=0.833)
4. kdtree — closeness=0.9571 (time=0.0713, memory=4998065, accuracy=0.833)
5. point_cloud — closeness=0.0425 (time=20.6357, memory=411424400, accuracy=1.000)
