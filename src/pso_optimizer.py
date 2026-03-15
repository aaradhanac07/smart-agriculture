import numpy as np
import json
import time
from multiprocessing import Pool, cpu_count

from src.parallel_eval import evaluate_rf_params
from src.config import BEST_PARAMS_PATH, RANDOM_STATE

class PSOOptimizer:
    def __init__(
        self,
        n_particles=10,
        n_iterations=10,
        w=0.7,
        c1=1.5,
        c2=1.5,
        n_estimators_bounds=(50, 400),
        max_depth_bounds=(3, 30),
    ):
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.n_estimators_bounds = n_estimators_bounds
        self.max_depth_bounds = max_depth_bounds

    def _clip(self, pos):
        pos[0] = np.clip(pos[0], *self.n_estimators_bounds)
        pos[1] = np.clip(pos[1], *self.max_depth_bounds)
        return pos

    def optimize(self, X_train, y_train, parallel=True):
        np.random.seed(RANDOM_STATE)

        # Particle positions: [n_estimators, max_depth]
        positions = np.zeros((self.n_particles, 2))
        velocities = np.zeros((self.n_particles, 2))

        for i in range(self.n_particles):
            positions[i, 0] = np.random.uniform(*self.n_estimators_bounds)
            positions[i, 1] = np.random.uniform(*self.max_depth_bounds)

        pbest_positions = positions.copy()
        pbest_scores = np.full(self.n_particles, -np.inf)

        gbest_position = None
        gbest_score = -np.inf

        start_time = time.time()

        for it in range(self.n_iterations):
            # Prepare evaluation args
            eval_args = []
            for i in range(self.n_particles):
                n_estimators = int(round(positions[i, 0]))
                max_depth = int(round(positions[i, 1]))
                eval_args.append((X_train, y_train, n_estimators, max_depth, RANDOM_STATE))

            # Evaluate fitness
            if parallel:
                with Pool(processes=min(cpu_count(), self.n_particles)) as pool:
                    scores = pool.map(evaluate_rf_params, eval_args)
            else:
                scores = [evaluate_rf_params(a) for a in eval_args]

            scores = np.array(scores)

            # Update personal best
            for i in range(self.n_particles):
                if scores[i] > pbest_scores[i]:
                    pbest_scores[i] = scores[i]
                    pbest_positions[i] = positions[i].copy()

            # Update global best
            best_idx = np.argmax(scores)
            if scores[best_idx] > gbest_score:
                gbest_score = scores[best_idx]
                gbest_position = positions[best_idx].copy()

            print(
                f"Iteration {it+1}/{self.n_iterations} | "
                f"Best CV Accuracy: {gbest_score:.4f} | "
                f"Best Params: n_estimators={int(round(gbest_position[0]))}, "
                f"max_depth={int(round(gbest_position[1]))}"
            )

            # Update velocities and positions
            for i in range(self.n_particles):
                r1, r2 = np.random.rand(2), np.random.rand(2)

                velocities[i] = (
                    self.w * velocities[i]
                    + self.c1 * r1 * (pbest_positions[i] - positions[i])
                    + self.c2 * r2 * (gbest_position - positions[i])
                )

                positions[i] = positions[i] + velocities[i]
                positions[i] = self._clip(positions[i])

        total_time = time.time() - start_time

        best_params = {
            "n_estimators": int(round(gbest_position[0])),
            "max_depth": int(round(gbest_position[1])),
            "best_cv_accuracy": float(gbest_score),
            "parallel": bool(parallel),
            "time_seconds": float(total_time),
        }

        with open(BEST_PARAMS_PATH, "w") as f:
            json.dump(best_params, f, indent=4)

        print("\n✅ PSO Optimization Completed!")
        print("💾 Best parameters saved to:", BEST_PARAMS_PATH)
        print("⏱️ Time taken:", round(total_time, 2), "seconds")

        return best_params
