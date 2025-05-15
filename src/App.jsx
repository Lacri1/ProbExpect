import { useState, useEffect } from "react";
import { Line } from "react-chartjs-2";
import {
    Chart as ChartJS,
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    Title,
    Tooltip,
    Legend,
} from "chart.js";
import "./App.css";

ChartJS.register(
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    Title,
    Tooltip,
    Legend
);

function App() {
    const [probPercent, setProbPercent] = useState(3);
    const [cost, setCost] = useState(270);
    const [batchSize, setBatchSize] = useState(1);
    const [data, setData] = useState(null);
    const [stats, setStats] = useState(null);
    const [dynamicMaxAttempts, setDynamicMaxAttempts] = useState(100);

    const p = probPercent / 100;
    const pBatch = 1 - Math.pow(1 - p, batchSize);
    // maxAttempts는 이제 동적으로 결정됩니다

    const findAttemptsForProb = (targetProb) => {
        const epsilon = 1e-9;
        for (let n = 1; n <= dynamicMaxAttempts; n++) {
            const cumProb = 1 - Math.pow(1 - pBatch, n);
            if (cumProb + epsilon >= targetProb) return n;
        }
        return dynamicMaxAttempts;
    };


    // 최적의 그래프 범위를 계산하는 함수
    const calculateOptimalAttempts = () => {
        // 확률에 따른 적응형 최대 시행 횟수 계산
        // 99%의 확률에 도달하는 횟수의 약 1.5배를 사용
        const attemptsFor99Percent = Math.ceil(Math.log(0.01) / Math.log(1 - pBatch));
        return Math.min(Math.max(Math.ceil(attemptsFor99Percent * 1.05), 20), 99900);
    };

    const handleCalculate = () => {
        if (p <= 0 || p > 1) {
            alert("확률은 0보다 크고 100 이하의 값이어야 합니다.");
            return;
        }
        if (cost <= 0) {
            alert("1회 비용은 0보다 커야 합니다.");
            return;
        }
        if (batchSize <= 0) {
            alert("1회 시도 시 뽑는 개수는 1 이상이어야 합니다.");
            return;
        }

        // 최적의 그래프 범위 계산
        const newMaxAttempts = calculateOptimalAttempts();
        setDynamicMaxAttempts(newMaxAttempts);

        const labels = [];
        const probData = [];
        for (let n = 1; n <= newMaxAttempts; n++) {
            labels.push(n);
            const cumProb = 1 - Math.pow(1 - pBatch, n);
            probData.push((cumProb * 100).toFixed(2));
        }

        setData({
            labels,
            datasets: [
                {
                    label: "1회 이상 당첨 확률 (%)",
                    data: probData,
                    borderColor: "rgba(75,192,192,1)",
                    backgroundColor: "rgba(75,192,192,0.2)",
                    fill: true,
                },
            ],
        });

        const n20 = findAttemptsForProb(0.2);
        const n63 = Math.ceil(Math.log(1 - 0.6321) / Math.log(1 - pBatch))
        const n80 = findAttemptsForProb(0.8);

        setStats({
            n20: { n: n20, cost: (n20 * cost).toLocaleString() },
            n63: { n: n63, cost: (n63 * cost).toLocaleString() },
            n80: { n: n80, cost: (n80 * cost).toLocaleString() },
            pBatchPercent: (pBatch * 100).toFixed(2),
            batchSize,
        });
    };

    // 컴포넌트가 처음 마운트될 때 기본값으로 계산 수행
    useEffect(() => {
        handleCalculate();
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);

    const options = {
        responsive: true,
        scales: {
            y: {
                min: 0,
                max: 100,
                title: { display: true, text: "확률 (%)" },
                ticks: { callback: (v) => `${v}%` },
            },
            x: {
                min: 1,
                max: dynamicMaxAttempts,
                title: { display: true, text: "시행 횟수 (n)" },
            },
        },
        plugins: {
            tooltip: {
                callbacks: {
                    title: (context) => `시행 횟수: ${context[0].label}`,
                    label: (context) => `확률: ${context.formattedValue}%`
                }
            }
        }
    };

    return (
        <div className="app-container">
            {/* 왼쪽에 그래프 영역 */}
            <div className="graph-section">
                <h1>기대값 계산기</h1>

                {data && (
                    <div className="chart-wrapper">
                        <Line data={data} options={options} />
                        <div className="chart-info">
                            (그래프는 확률 {probPercent}%에 최적화된 {dynamicMaxAttempts}회 시행까지 표시됩니다)
                        </div>
                    </div>
                )}
            </div>

            {/* 오른쪽에 컨트롤 및 통계 영역 */}
            <div className="right-panel">
                <div className="input-controls">
                    <div className="input-group">
                        <label htmlFor="prob-percent">당첨 확률 (%)</label>
                        <input
                            id="prob-percent"
                            type="number"
                            value={probPercent}
                            min="0"
                            max="100"
                            step="0.01"
                            onChange={(e) => setProbPercent(Number(e.target.value))}
                            className="input-number"
                        />
                    </div>

                    <div className="input-group">
                        <label htmlFor="cost">1회 비용</label>
                        <input
                            id="cost"
                            type="number"
                            value={cost}
                            min="0"
                            onChange={(e) => setCost(Number(e.target.value))}
                            className="input-number"
                        />
                    </div>

                    <div className="input-group">
                        <label htmlFor="batch-size">시행 당 뽑는 개수</label>
                        <input
                            id="batch-size"
                            type="number"
                            value={batchSize}
                            min="1"
                            onChange={(e) => setBatchSize(Number(e.target.value))}
                            className="input-number"
                        />
                    </div>

                    <button onClick={handleCalculate} className="btn-calc">
                        계산하기
                    </button>
                </div>

                {stats && (
                    <div className="stats-section">
                        <h2>요구 시행횟수 및 비용</h2>

                        <div className="batch-info">
                            <p>
                                <strong>1회 시도 시 뽑는 개수:</strong> {stats.batchSize}개
                            </p>
                            <p>
                                <strong>시행당 당첨 확률:</strong> {stats.pBatchPercent}%
                            </p>
                        </div>

                        <ul className="stats-list">
                            <li>
                                <strong>상위 20%:</strong>
                                <span className="stats-result">
                                    {stats.n20.n}회 시행,
                                    <span className="stats-cost"> 총 비용 {stats.n20.cost}</span>
                                </span>
                            </li>
                            <li>
                                <strong>평균 63.21%:</strong>
                                <span className="stats-result">
                                    {stats.n63.n}회 시행,
                                    <span className="stats-cost"> 총 비용 {stats.n63.cost}</span>
                                </span>
                            </li>
                            <li>
                                <strong>상위 80%:</strong>
                                <span className="stats-result">
                                    {stats.n80.n}회 시행,
                                    <span className="stats-cost"> 총 비용 {stats.n80.cost}</span>
                                </span>
                            </li>
                        </ul>
                    </div>
                )}
            </div>
        </div>
    );
}

export default App;
