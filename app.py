import os
from flask import Flask, render_template, request, jsonify
import numpy as np

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/calculate', methods=['POST'])
def calculate():
    data = request.json
    n = int(data['n'])
    start = data['start']  # [r, c]
    end = data['end']      # [r, c]
    walls = data['walls']  # [[r, c], ...]
    mode = data['mode']    # 'random' 或 'optimal'

    gamma = 0.9
    v_matrix = np.zeros((n, n))
    policy = [["" for _ in range(n)] for _ in range(n)]
    
    # 行動定義: 0:上, 1:下, 2:左, 3:右
    actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    arrows = ["↑", "↓", "←", "→"]

    # 簡單起見：每走一步 reward -1，到達終點 reward 0
    def get_reward(r, c):
        if [r, c] == end: return 0
        return -1

    if mode == 'random':
        # HW1-2: 隨機策略評估 (Iterative Policy Evaluation)
        # 隨機生成策略
        for r in range(n):
            for c in range(n):
                if [r, c] == end or [r, c] in walls: continue
                policy[r][c] = arrows[np.random.randint(4)]
        
        # 進行 100 次迭代更新價值
        for _ in range(100):
            new_v = v_matrix.copy()
            for r in range(n):
                for c in range(n):
                    if [r, c] == end or [r, c] in walls: continue
                    # 隨機策略下，每個方向機率 0.25
                    v_sum = 0
                    for dr, dc in actions:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < n and 0 <= nc < n and [nr, nc] not in walls:
                            v_sum += 0.25 * (get_reward(nr, nc) + gamma * v_matrix[nr][nc])
                        else:
                            v_sum += 0.25 * (get_reward(r, c) + gamma * v_matrix[r][c])
                    new_v[r][c] = v_sum
            v_matrix = new_v

    else:
        # HW1-3: 價值迭代 (Value Iteration)
        for _ in range(100):
            new_v = v_matrix.copy()
            for r in range(n):
                for c in range(n):
                    if [r, c] == end or [r, c] in walls: continue
                    
                    q_values = []
                    for i, (dr, dc) in enumerate(actions):
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < n and 0 <= nc < n and [nr, nc] not in walls:
                            q_values.append(get_reward(nr, nc) + gamma * v_matrix[nr][nc])
                        else:
                            q_values.append(get_reward(r, c) + gamma * v_matrix[r][c])
                    
                    new_v[r][c] = max(q_values)
                    policy[r][c] = arrows[np.argmax(q_values)]
            v_matrix = new_v

    return jsonify({
        'v_matrix': np.round(v_matrix, 2).tolist(),
        'policy': policy
    })

if __name__ == '__main__':
    # Render 需要監聽 0.0.0.0
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
