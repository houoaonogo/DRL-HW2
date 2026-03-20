import os
from flask import Flask, render_template, request, jsonify
import numpy as np
import random

# 修改這行
app = Flask(__name__, template_folder='../templates')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/calculate', methods=['POST'])
def calculate():
    data = request.json
    n = int(data['n'])
    start = data['start'] # [r, c]
    end = data['end']     # [r, c]
    walls = data['walls'] # [[r, c], ...]
    mode = data['mode']   # 'hw1-2' (random) or 'hw1-3' (optimal)

    gamma = 0.9
    v_matrix = np.zeros((n, n))
    policy = [["" for _ in range(n)] for _ in range(n)]
    
    # 0:上, 1:下, 2:左, 3:右
    actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    arrows = ["↑", "↓", "←", "→"]

    if mode == 'hw1-2':
        # --- HW1-2: 隨機策略與價值評估 ---
        # 1. 生成隨機策略
        for r in range(n):
            for c in range(n):
                if [r, c] == end or [r, c] in walls: continue
                policy[r][c] = random.choice(arrows)
        
        # 2. 策略評估 (Iterative Policy Evaluation)
        for _ in range(200):
            new_v = v_matrix.copy()
            for r in range(n):
                for c in range(n):
                    if [r, c] == end or [r, c] in walls: continue
                    
                    # 隨機行動 (每個方向機率 0.25)
                    v_sum = 0
                    for dr, dc in actions:
                        nr, nc = r + dr, c + dc
                        reward = 10 if [nr, nc] == end else -1
                        if 0 <= nr < n and 0 <= nc < n and [nr, nc] not in walls:
                            v_sum += 0.25 * (reward + gamma * v_matrix[nr][nc])
                        else:
                            v_sum += 0.25 * (-1 + gamma * v_matrix[r][c]) # 撞牆
                    new_v[r][c] = v_sum
            v_matrix = new_v

    else:
        # --- HW1-3: 價值迭代與最佳策略 ---
        for _ in range(200):
            new_v = v_matrix.copy()
            for r in range(n):
                for c in range(n):
                    if [r, c] == end or [r, c] in walls: continue
                    
                    qs = []
                    for dr, dc in actions:
                        nr, nc = r + dr, c + dc
                        reward = 10 if [nr, nc] == end else -1
                        if 0 <= nr < n and 0 <= nc < n and [nr, nc] not in walls:
                            qs.append(reward + gamma * v_matrix[nr][nc])
                        else:
                            qs.append(-1 + gamma * v_matrix[r][c])
                    
                    new_v[r][c] = max(qs)
                    policy[r][c] = arrows[np.argmax(qs)]
            v_matrix = new_v

    return jsonify({
        'v_matrix': np.round(v_matrix, 2).tolist(),
        'policy': policy
    })

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    #app.run(host='0.0.0.0', port=port)
