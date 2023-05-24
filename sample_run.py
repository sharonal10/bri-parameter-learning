from pipeline import *

normal_dist_preds = generate_samples(run(sys.argv[1], sys.argv[2]))
print(normal_dist_preds)

print('Predicted Coordinate:', normal_dist_preds[0])