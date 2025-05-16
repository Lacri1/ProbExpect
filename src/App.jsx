import { useState, useEffect, useMemo, useCallback } from "react";
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

// 로그팩토리얼 캐싱 테이블
const LOG_FACTORIAL_CACHE = [0]; // log(0!) = log(1) = 0

// 로그팩토리얼 계산 (큰 숫자 처리를 위해 로그 영역에서 계산)
const logFactorial = (n) => {
    // 캐시에 없는 값들을 계산하여 추가
    if (LOG_FACTORIAL_CACHE.length <= n) {
        for (let i = LOG_FACTORIAL_CACHE.length; i <= n; i++) {
            LOG_FACTORIAL_CACHE[i] = LOG_FACTORIAL_CACHE[i - 1] + Math.log(i);
        }
    }
    return LOG_FACTORIAL_CACHE[n];
};

// 로그 영역에서 이항계수 계산
const logCombination = (n, k) => {
    if (k < 0 || k > n) return -Infinity; // 불가능한 경우
    if (k === 0 || k === n) return 0; // log(1) = 0
    return logFactorial(n) - logFactorial(k) - logFactorial(n - k);
};

// 로그 영역에서 이항확률 계산
const logBinomialProbability = (n, k, p) => {
    if (p === 0) return k === 0 ? 0 : -Infinity;
    if (p === 1) return k === n ? 0 : -Infinity;

    const logComb = logCombination(n, k);
    return logComb + k * Math.log(p) + (n - k) * Math.log(1 - p);
};

// 이항 확률 P(X = k)
const binomialProbability = (n, k, p) => {
    if (p === 0) return k === 0 ? 1 : 0;
    if (p === 1) return k === n ? 1 : 0;

    // 작은 n, k에 대해서는 직접 계산이 더 효율적
    if (n < 100) {
        const comb = Math.exp(logCombination(n, k));
        return comb * Math.pow(p, k) * Math.pow(1 - p, n - k);
    }

    // 큰 수에 대해서는 로그 영역에서 계산
    return Math.exp(logBinomialProbability(n, k, p));
};

// 이항 분포 누적 확률: P(X >= k) - 더 효율적인 방법
const cumulativeProbabilityAtLeastK = (n, k, p) => {
    // 경계 케이스 처리
    if (k <= 0) return 1;
    if (k > n) return 0;

    // p가 매우 작고 k가 크면 근사치 사용
    if (p < 0.001 && k > 50) {
        const mean = n * p;
        const stdDev = Math.sqrt(n * p * (1 - p));
        // 정규분포 근사
        return 1 - cumulativeNormal((k - 0.5 - mean) / stdDev);
    }

    // k가 n의 절반 이상이면 반대 방향으로 계산하는 것이 더 효율적
    if (k > n / 2 + 1) {
        let sum = 0;
        for (let i = 0; i < k; i++) {
            sum += binomialProbability(n, i, p);
        }
        return 1 - sum;
    }

    // 일반적인 경우: P(X >= k) 직접 계산
    let sum = 0;
    for (let i = k; i <= n; i++) {
        const binomProb = binomialProbability(n, i, p);
        sum += binomProb;

        // 합이 충분히 1에 가까워지면 조기 종료
        if (sum > 0.9999) {
            return sum;
        }
    }
    return sum;
};

// 표준정규분포 누적분포함수 (CDF) 근사
function cumulativeNormal(x) {
    const a1 = 0.254829592;
    const a2 = -0.284496736;
    const a3 = 1.421413741;
    const a4 = -1.453152027;
    const a5 = 1.061405429;
    const p = 0.3275911;

    const sign = x < 0 ? -1 : 1;
    x = Math.abs(x) / Math.sqrt(2.0);

    const t = 1.0 / (1.0 + p * x);
    const erf = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);

    return 0.5 * (1.0 + sign * erf);
}

// 직접 확률 계산을 피하기 위한 최적화된 함수
const findAttemptsForProbOptimized = (targetProb, p, batchSize, isMultipleWin, targetWinCount, maxAttempts) => {
    const epsilon = 1e-6;
    const pBatch = 1 - Math.pow(1 - p, batchSize);

    // 단순한 경우: 1회 이상 당첨
    if (!isMultipleWin || targetWinCount === 1) {
        // log(1-P) 영역에서 계산
        const logOneMinusP = Math.log(1 - pBatch);
        const logOneMinusTarget = Math.log(1 - targetProb);

        // n = log(1-target) / log(1-p)
        const n = Math.ceil(logOneMinusTarget / logOneMinusP);
        return Math.min(n, maxAttempts);
    }

    // 평균(λ)이 작은 경우, 이항분포 대신 포아송 분포로 근사
    if (pBatch < 0.01 && maxAttempts * pBatch < 10) {
        const lambda = maxAttempts * pBatch; // 포아송 평균
        let poissProb = 0;
        let k = targetWinCount - 1;

        // P(X >= targetWinCount) = 1 - P(X < targetWinCount)
        while (k >= 0) {
            poissProb += Math.exp(-lambda) * Math.pow(lambda, k) / factorial(k);
            k--;
        }

        return Math.ceil(-Math.log(poissProb) / pBatch);
    }

    // 이분 탐색으로 필요한 시행 횟수 찾기
    let left = 1;
    let right = maxAttempts;
    let result = maxAttempts;

    while (left <= right) {
        const mid = Math.floor((left + right) / 2);
        let cumProb;

        if (isMultipleWin) {
            cumProb = cumulativeProbabilityAtLeastK(mid, targetWinCount, pBatch);
        } else {
            cumProb = 1 - Math.pow(1 - pBatch, mid);
        }

        if (cumProb + epsilon >= targetProb) {
            result = mid;
            right = mid - 1;
        } else {
            left = mid + 1;
        }
    }

    return result;
};

// 작은 정수 팩토리얼 계산
function factorial(n) {
    if (n <= 1) return 1;
    let result = 1;
    for (let i = 2; i <= n; i++) {
        result *= i;
    }
    return result;
}

function App() {
    const [probPercent, setProbPercent] = useState("3");
    const [cost, setCost] = useState("270");
    const [batchSize, setBatchSize] = useState("1");
    const [data, setData] = useState(null);
    const [stats, setStats] = useState(null);
    const [dynamicMaxAttempts, setDynamicMaxAttempts] = useState(100);
    const [isMultipleWin, setIsMultipleWin] = useState(false);
    const [targetWinCount, setTargetWinCount] = useState("1");
    const [calculating, setCalculating] = useState(false);

    const p = useMemo(() => probPercent / 100, [probPercent]);
    const pBatch = useMemo(() => 1 - Math.pow(1 - p, batchSize), [p, batchSize]);

    // 계산이 무거울 때 사용할 제한값
    const MAX_SAFE_COMPUTATION = 10000;
    const COMPUTATION_WARNING_THRESHOLD = 5000;

    // 안전한 계산 체크
    const isHeavyComputation = useCallback(() => {
        // 매우 낮은 확률에 높은 목표 당첨 횟수
        return isMultipleWin &&
            targetWinCount > 10 &&
            pBatch < 0.01 &&
            (targetWinCount / pBatch) > COMPUTATION_WARNING_THRESHOLD;
    }, [isMultipleWin, targetWinCount, pBatch]);

    const calculateOptimalAttempts = useCallback(() => {
        if (isHeavyComputation()) {
            return MAX_SAFE_COMPUTATION;
        }

        // 모든 경우에 공통으로 95% 확률 계산한 뒤 1.2배
        const n95 = findAttemptsForProbOptimized(0.95, p, batchSize, isMultipleWin, targetWinCount, 99900);
        return Math.min(Math.ceil(n95 * 1.2), 99900);
    }, [p, batchSize, isMultipleWin, targetWinCount]);

    // 계산 함수
    const handleCalculate = useCallback(() => {
        // 문자열 상태를 숫자로 변환
        const prob = Number(probPercent);
        const c = Number(cost);
        const batch = Number(batchSize);
        const target = Number(targetWinCount);

        if (
            isNaN(prob) || prob <= 0 || prob > 100 ||
            isNaN(c) || c <= 0 ||
            isNaN(batch) || batch <= 0 ||
            (isMultipleWin && (isNaN(target) || target < 1))
        ) {
            alert("모든 입력란에 올바른 숫자를 입력해주세요.");
            return;
        }

        // 계산 시작
        setCalculating(true);


        // 비동기로 처리하여 UI 블록 방지
        setTimeout(() => {
            try {
                // 계산량이 과도하게 큰 경우 경고
                if (isHeavyComputation()) {
                    alert("계산량이 매우 큽니다. 결과가 근사치로 제공되거나 일부 데이터가 제한될 수 있습니다.");
                }

                const newMaxAttempts = calculateOptimalAttempts();
                setDynamicMaxAttempts(newMaxAttempts);

                const samplePoints = Math.min(200, newMaxAttempts); // 그래프 데이터 포인트 제한
                const step = Math.max(1, Math.floor(newMaxAttempts / samplePoints));

                const labels = [];
                const probData = [];

                // 그래프 데이터 포인트 샘플링
                for (let n = 1; n <= newMaxAttempts; n += step) {
                    let prob;
                    if (isMultipleWin) {
                        prob = cumulativeProbabilityAtLeastK(n, targetWinCount, pBatch);
                    } else {
                        prob = 1 - Math.pow(1 - pBatch, n);
                    }
                    labels.push(n);
                    probData.push((prob * 100).toFixed(2));
                }

                // 마지막 포인트 추가 (최대값)
                if (labels[labels.length - 1] !== newMaxAttempts) {
                    let prob;
                    if (isMultipleWin) {
                        prob = cumulativeProbabilityAtLeastK(newMaxAttempts, targetWinCount, pBatch);
                    } else {
                        prob = 1 - Math.pow(1 - pBatch, newMaxAttempts);
                    }
                    labels.push(newMaxAttempts);
                    probData.push((prob * 100).toFixed(2));
                }

                // 주요 통계치 계산
                const n20 = findAttemptsForProbOptimized(0.2, p, batchSize, isMultipleWin, targetWinCount, newMaxAttempts);
                const n63 = findAttemptsForProbOptimized(0.6321, p, batchSize, isMultipleWin, targetWinCount, newMaxAttempts);
                const n80 = findAttemptsForProbOptimized(0.8, p, batchSize, isMultipleWin, targetWinCount, newMaxAttempts);

                setStats({
                    n20: { n: n20, cost: (n20 * cost).toLocaleString() },
                    n63: { n: n63, cost: (n63 * cost).toLocaleString() },
                    n80: { n: n80, cost: (n80 * cost).toLocaleString() },
                    pBatchPercent: (pBatch * 100).toFixed(2),
                    batchSize,
                });

                setData({
                    labels,
                    datasets: [
                        {
                            label: isMultipleWin
                                ? `${targetWinCount}회 이상 당첨 확률 (%)`
                                : "1회 이상 당첨 확률 (%)",
                            data: probData,
                            borderColor: "rgba(75,192,192,1)",
                            backgroundColor: "rgba(75,192,192,0.2)",
                            fill: true,
                        },
                    ],
                });
            } catch (error) {
                console.error("계산 중 오류 발생:", error);
                alert("계산 중 오류가 발생했습니다. 입력값을 조정해보세요.");
            } finally {
                setCalculating(false);
            }
        }, 100);
    }, [probPercent, cost, batchSize, targetWinCount, isMultipleWin]);

    // 자동 계산 제거 (useEffect 제거)

    // 페이지 로드 시 초기 계산 수행
    useEffect(() => {
        handleCalculate();
    }, []); // 컴포넌트 마운트 시 한 번만 실행

    const chartOptions = useMemo(() => ({
        responsive: true,
        maintainAspectRatio: false,
        animation: {
            duration: 1000 // 애니메이션 활성화 (1초)
        },
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
    }), [dynamicMaxAttempts]);

    return (
        <div className="app-container">
            <div className="graph-section">
                <h1>기댓값 계산기</h1>
                {data && (
                    <div className="chart-wrapper">
                        <Line data={data} options={chartOptions} />
                        <div className="chart-info">
                            (그래프는 {isMultipleWin ? `목표 ${targetWinCount}회` : `확률 ${probPercent}%`}에 최적화된 {dynamicMaxAttempts}회 시행까지 표시됩니다)
                        </div>
                    </div>
                )}
            </div>

            <div className="right-panel">
                <div className="input-controls">
                    <div className="input-group">
                        <label htmlFor="prob-percent">당첨 확률 (%)</label>
                        <input
                            id="prob-percent"
                            type="text"
                            value={probPercent}
                            onChange={(e) => setProbPercent(e.target.value)}
                            className="input-number"
                            disabled={calculating}
                        />
                    </div>

                    <div className="input-group">
                        <label htmlFor="cost">1회 비용</label>
                        <input
                            id="cost"
                            type="text"
                            value={cost}
                            onChange={(e) => setCost(e.target.value)}
                            className="input-number"
                            disabled={calculating}
                        />
                    </div>

                    <div className="input-group">
                        <label htmlFor="batch-size">시행 당 뽑는 개수</label>
                        <input
                            id="batch-size"
                            type="text"
                            value={batchSize}
                            onChange={(e) => setBatchSize(e.target.value)}
                            className="input-number"
                            disabled={calculating}
                        />
                    </div>

                    <div className="input-group">
                        <label>
                            <input
                                type="checkbox"
                                checked={isMultipleWin}
                                onChange={() => setIsMultipleWin(!isMultipleWin)}
                                disabled={calculating}
                            />
                            목표 당첨 횟수
                        </label>
                    </div>

                    {isMultipleWin && (
                        <div className="input-group">
                            <label htmlFor="target-win-count">목표 당첨 횟수</label>
                            <input
                                id="target-win-count"
                                type="text"
                                value={targetWinCount}
                                onChange={(e) => setTargetWinCount(e.target.value)}
                                className="input-number"
                                disabled={calculating}
                            />
                        </div>
                    )}

                    <button onClick={handleCalculate} className="btn-calc" disabled={calculating}>
                        {calculating ? "계산 중..." : "계산하기"}
                    </button>
                </div>

                {stats && (
                    <div className="stats-section">
                        <h2>요구 시행횟수 및 비용</h2>
                        <div className="batch-info">
                            <p><strong>1회 시도 시 뽑는 개수:</strong> {stats.batchSize}개</p>
                            <p><strong>시행당 당첨 확률:</strong> {stats.pBatchPercent}%</p>
                        </div>
                        <ul className="stats-list">
                            <li>상위 20% <strong>{stats.n20.n}회, 총 {stats.n20.cost}</strong></li>
                            <li>평균 63.2% <strong>{stats.n63.n}회, 총 {stats.n63.cost}</strong></li>
                            <li>상위 80% <strong>{stats.n80.n}회, 총 {stats.n80.cost}</strong></li>
                        </ul>
                    </div>
                )}
            </div>
        </div>
    );
}

export default App;