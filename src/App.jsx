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

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend);

// 로그팩토리얼 캐시
const LOG_FACTORIAL_CACHE = [0];
const logFactorial = (n) => {
    if (LOG_FACTORIAL_CACHE.length <= n) {
        for (let i = LOG_FACTORIAL_CACHE.length; i <= n; i++) {
            LOG_FACTORIAL_CACHE[i] = LOG_FACTORIAL_CACHE[i - 1] + Math.log(i);
        }
    }
    return LOG_FACTORIAL_CACHE[n];
};

const logCombination = (n, k) => {
    if (k < 0 || k > n) return -Infinity;
    if (k === 0 || k === n) return 0;
    return logFactorial(n) - logFactorial(k) - logFactorial(n - k);
};

const logBinomialProbability = (n, k, p) => {
    if (p === 0) return k === 0 ? 0 : -Infinity;
    if (p === 1) return k === n ? 0 : -Infinity;
    const logComb = logCombination(n, k);
    const kLogP = k > 0 ? k * Math.log(p) : 0;
    const nMinusKLogOneMinusP = (n > k && p < 0.9999999999) ? (n - k) * Math.log(1 - p) : -Infinity;
    return logComb + kLogP + nMinusKLogOneMinusP;
};

const binomialProbability = (n, k, p) => {
    if (p === 0) return k === 0 ? 1 : 0;
    if (p === 1) return k === n ? 1 : 0;
    if (p < 1e-15 && k > 0) return 0;
    if (p > 0.9999999999 && k < n) return 0;
    if (n < 100) {
        try {
            const comb = Math.exp(logCombination(n, k));
            const result = comb * Math.pow(p, k) * Math.pow(1 - p, n - k);
            return isNaN(result) ? 0 : result;
        } catch (e) {
            console.error('binomialProbability error:', e);
            return 0;
        }
    }
    const logProb = logBinomialProbability(n, k, p);
    if (logProb === -Infinity || isNaN(logProb) || logProb < -700) return 0;
    const result = Math.exp(logProb);
    return isNaN(result) ? 0 : result;
};

const cumulativeNormal = (x) => {
    const a1 = 0.254829592, a2 = -0.284496736, a3 = 1.421413741;
    const a4 = -1.453152027, a5 = 1.061405429, p = 0.3275911;
    const sign = x < 0 ? -1 : 1;
    x = Math.abs(x) / Math.sqrt(2.0);
    const t = 1.0 / (1.0 + p * x);
    const erf = 1.0 - (((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t) * Math.exp(-x * x);
    return 0.5 * (1.0 + sign * erf);
};

const cumulativeProbabilityAtLeastK = (n, k, p) => {
    if (k <= 0) return 1;
    if (k > n) return 0;
    if (p < 1e-10 && k > 1) return 0;
    if (p > 0.9999999999) return 1;

    if (p < 0.001 && k > 50) {
        const mean = n * p;
        const stdDev = Math.sqrt(n * p * (1 - p));
        const z = (k - 0.5 - mean) / stdDev;
        const approx = 1 - cumulativeNormal(z);
        if (approx <= 0) {
            let sum = 0;
            for (let i = k; i <= Math.min(n, k + 100); i++) {
                const bp = binomialProbability(n, i, p);
                if (!isNaN(bp)) sum += bp;
            }
            return sum;
        }
        return approx;
    }

    if (k > n / 2 + 1) {
        let sum = 0;
        for (let i = 0; i < k; i++) {
            const bp = binomialProbability(n, i, p);
            if (!isNaN(bp)) sum += bp;
        }
        return Math.max(0, Math.min(1, 1 - sum));
    }

    let sum = 0;
    for (let i = k; i <= n; i++) {
        const bp = binomialProbability(n, i, p);
        if (!isNaN(bp)) sum += bp;
        if (sum > 0.9999) break;
    }
    return isNaN(sum) ? 0 : Math.max(0, Math.min(1, sum));
};

// 복수 성공을 처리하기 위한 확률 계산 (목표 성공 횟수까지)
const calculateProbabilityForMultipleWins = (numBatches, p, batchSize, targetWinCount) => {
    // 각 시행에서 개별 확률 p로 batchSize번 시도
    // 전체 성공 횟수가 targetWinCount 이상일 확률 계산

    // 각 배치에서 성공할 확률은 여전히 p
    // 총 시행 횟수 = numBatches * batchSize
    // 목표 성공 횟수 = targetWinCount

    const totalTrials = numBatches * batchSize;
    return cumulativeProbabilityAtLeastK(totalTrials, targetWinCount, p);
};

// 초저확률 상황에서 근사적 계산을 위한 함수
const estimateAttemptsForVeryLowProbability = (targetProb, p, batchSize) => {
    // 매우 작은 확률에서는 포아송 근사 사용
    // 배치당 성공 확률(pBatch)이 작고, 시행횟수(n)가 크면 포아송 분포로 근사 가능
    const pBatch = 1 - Math.pow(1 - p, batchSize);

    // 포아송 분포의 λ 파라미터 계산: λ = n * pBatch
    // targetProb = 1 - e^(-λ)에서 λ 계산
    const lambda = -Math.log(1 - targetProb);

    // n = λ / pBatch
    return Math.ceil(lambda / pBatch);
};

// 직접 확률 계산을 피하기 위한 최적화된 함수
const findAttemptsForProbOptimized = (
    targetProb,
    p,
    batchSize,
    isMultipleWin,
    targetWinCount,
    maxAttempts
) => {
    const epsilon = 1e-10;

    // 단순한 경우: 1회 이상 당첨 (복수 시행 당첨 미적용)
    if (!isMultipleWin || targetWinCount === 1) {
        if (p < 0.0001) {
            return estimateAttemptsForVeryLowProbability(targetProb, p, batchSize);
        }

        const pBatch = 1 - Math.pow(1 - p, batchSize);
        if (pBatch < 1e-10) {
            return estimateAttemptsForVeryLowProbability(targetProb, p, batchSize);
        }

        const logOneMinusP = Math.log(1 - pBatch);
        const logOneMinusTarget = Math.log(1 - targetProb);
        const n = Math.ceil(logOneMinusTarget / logOneMinusP);
        return Math.min(n, maxAttempts);
    }

    // 복수 당첨 이분 탐색
    const lowerBound = 1;
    let upperBound;

    if (p >= 0.001) {
        // 단순한 경우 시행 횟수 추정
        const pBatch = 1 - Math.pow(1 - p, batchSize);
        const logOneMinusP = Math.log(1 - pBatch);
        const logOneMinusTarget = Math.log(1 - targetProb);
        const simpleN = Math.ceil(logOneMinusTarget / logOneMinusP);

        // 목표 당첨 횟수를 곱하여 upperBound 설정
        upperBound = Math.min(maxAttempts, simpleN * targetWinCount);
    } else {
        // 저확률일수록 여유 있게 scaleFactor를 사용
        const scaleFactor = p < 0.0001 ? 100 : (p < 0.001 ? 10 : 1);
        upperBound = Math.ceil(Math.min(maxAttempts, scaleFactor * targetWinCount / (p * batchSize)));
    }

    let left = lowerBound;
    let right = upperBound;
    let result = upperBound;

    while (left <= right) {
        const mid = Math.floor((left + right) / 2);
        const prob = calculateProbabilityForMultipleWins(mid, p, batchSize, targetWinCount);

        if (prob >= targetProb - epsilon) {
            result = mid;
            right = mid - 1;
        } else {
            left = mid + 1;
        }
    }

    return Math.min(result, maxAttempts);
};


// 숫자를 적절한 형식으로 포맷팅하는 함수
const formatNumber = (num) => {
    if (num >= 1e6) {
        // 백만 이상은 지수 표기법 사용
        return num.toExponential(2).replace(/e\+?/, ' × 10^');
    }
    return num.toLocaleString();
};

function App() {
    const [probPercent, setProbPercent] = useState("3");  // 기본값
    const [cost, setCost] = useState("2700");
    const [batchSize, setBatchSize] = useState("10");
    const [data, setData] = useState(null);
    const [stats, setStats] = useState(null);
    const [dynamicMaxAttempts, setDynamicMaxAttempts] = useState(100000);  // 기본 최대 시도 횟수 증가
    const [isMultipleWin, setIsMultipleWin] = useState(false);
    const [targetWinCount, setTargetWinCount] = useState("10");
    const [calculating, setCalculating] = useState(false);
    const [chartInfo, setChartInfo] = useState(null); // 차트 정보 상태 추가

    // 입력된 확률을 소수점으로 변환
    const p = useMemo(() => {
        // 입력값을 항상 비율로 처리
        return Number(probPercent) / 100;
    }, [probPercent]);

    // 계산이 무거울 때 사용할 제한값
    const MAX_SAFE_COMPUTATION = 1000000000;  // 최대 계산 범위 증가
    const COMPUTATION_WARNING_THRESHOLD = 10000;

    // 안전한 계산 체크
    const isHeavyComputation = useCallback(() => {
        // 매우 낮은 확률에 높은 목표 당첨 횟수
        return isMultipleWin &&
            targetWinCount > 100 &&
            p < 0.01 &&
            (targetWinCount / p) > COMPUTATION_WARNING_THRESHOLD;
    }, [isMultipleWin, targetWinCount, p]);

    const calculateOptimalAttempts = useCallback(() => {
        if (isHeavyComputation()) {
            return MAX_SAFE_COMPUTATION;
        }

        // 매우 낮은 확률에 대한 특별 처리
        if (p < 0.0001) {
            // 단일 당첨 모델
            if (!isMultipleWin) {
                // 기대값의 약 10배 (더 넓은 범위 확보)
                return Math.min(Math.ceil(10 / (p * batchSize)), MAX_SAFE_COMPUTATION);
            } else {
                // 복수 당첨 모델
                // 목표 성공 횟수를 달성하기 위한 기대 시도 횟수의 약 10배
                return Math.min(Math.ceil(10 * targetWinCount / (p * batchSize)), MAX_SAFE_COMPUTATION);
            }
        }

        // 모든 경우에 공통으로 99% 확률을 충분히 보기 위한 범위 사용
        const n99 = findAttemptsForProbOptimized(0.99, p, batchSize, isMultipleWin, targetWinCount, MAX_SAFE_COMPUTATION);
        return Math.min(Math.ceil(n99 * 1.5), MAX_SAFE_COMPUTATION);
    }, [p, batchSize, isMultipleWin, targetWinCount]);

    // 계산 함수
    const handleCalculate = useCallback(() => {
        // 문자열 상태를 숫자로 변환
        const inputProb = Number(probPercent);
        const c = Number(cost);
        const batch = Number(batchSize);
        const target = Number(targetWinCount);

        // 유효성 검사
        if (
            isNaN(inputProb) || inputProb <= 0 ||
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

                // 최적 시도 횟수 계산 (배치 단위)
                const newMaxBatches = calculateOptimalAttempts();
                setDynamicMaxAttempts(newMaxBatches);

                // 그래프 데이터 포인트 수 결정 (너무 많은 포인트는 성능에 영향)
                const maxDataPoints = 100;
                const step = Math.max(1, Math.floor(newMaxBatches / maxDataPoints));

                const labels = [];
                const probData = [];
                let maxProbForScaling = 0; // 최대 확률값 추적

                // 그래프 데이터 포인트 샘플링 (배치 단위로)
                for (let n = 1; n <= newMaxBatches; n += step) {
                    let prob;
                    if (isMultipleWin) {
                        // 복수 당첨 모델: n개 배치에서 targetWinCount 이상 성공할 확률
                        prob = calculateProbabilityForMultipleWins(n, p, batch, target);
                    } else {
                        // 단일 당첨 모델 (기존 방식)
                        const pBatch = 1 - Math.pow(1 - p, batch);
                        prob = 1 - Math.pow(1 - pBatch, n);
                    }

                    // NaN 값 방지
                    if (isNaN(prob)) {
                        prob = 0;
                    }

                    // 최대 확률 업데이트
                    maxProbForScaling = Math.max(maxProbForScaling, prob);

                    labels.push(n);
                    probData.push((prob * 100).toFixed(4)); // 정밀도 증가
                }

                // 마지막 포인트 추가 (최대값)
                if (labels[labels.length - 1] !== newMaxBatches) {
                    let prob;
                    if (isMultipleWin) {
                        prob = calculateProbabilityForMultipleWins(newMaxBatches, p, batch, target);
                    } else {
                        const pBatch = 1 - Math.pow(1 - p, batch);
                        prob = 1 - Math.pow(1 - pBatch, newMaxBatches);
                    }

                    // NaN 값 방지
                    if (isNaN(prob)) {
                        prob = 0;
                    }

                    // 최대 확률 업데이트
                    maxProbForScaling = Math.max(maxProbForScaling, prob);

                    labels.push(newMaxBatches);
                    probData.push((prob * 100).toFixed(4)); // 정밀도 증가
                }

                // 확률이 매우 낮을 때 적응형 y축 스케일 추가
                const maxYScale = maxProbForScaling < 0.1
                    ? Math.ceil(maxProbForScaling * 100 * 1.2)
                    : 100;


                // 주요 통계치 계산
                const n20 = findAttemptsForProbOptimized(0.2, p, batch, isMultipleWin, target, newMaxBatches);
                const n63 = findAttemptsForProbOptimized(0.6321, p, batch, isMultipleWin, target, newMaxBatches);
                const n80 = findAttemptsForProbOptimized(0.8, p, batch, isMultipleWin, target, newMaxBatches);

                // 차트 정보 텍스트 업데이트
                setChartInfo({
                    isMultipleWin,
                    targetWinCount: target,
                    probability: (() => {
                        const raw = p * 100;
                        let fixed =
                            raw < 0.001 ? raw.toFixed(6) :
                                raw < 0.01 ? raw.toFixed(4) :
                                    raw.toFixed(2);
                        return fixed.replace(/\\.?(0)+$/, "");  // 소수점 이하 0 제거
                    })(),
                    dynamicMaxAttempts: newMaxBatches,
                    batchSize: batch,
                    maxYScale
                });

                // 통계 설정
                setStats({
                    n20: { n: n20, cost: (n20 * c).toLocaleString() },
                    n63: { n: n63, cost: (n63 * c).toLocaleString() },
                    n80: { n: n80, cost: (n80 * c).toLocaleString() },
                    pIndividual: p, // 개별 시행당 당첨 확률
                    pPercent: ((str => str.replace(/\.?0+$/, ""))((p * 100).toFixed(p < 0.01 ? 6 : 2))),
                    batchSize: batch,
                    totalTrials: {
                        n20: n20 * batch,
                        n63: n63 * batch,
                        n80: n80 * batch
                    }
                });

                setData({
                    labels,
                    datasets: [
                        {
                            label: isMultipleWin
                                ? `${target}회 이상 당첨 확률 (%)`
                                : "당첨 확률 (%)",
                            data: probData,
                            borderColor: "rgba(75,192,192,1)",
                            backgroundColor: "rgba(75,192,192,0.2)",
                            fill: true,
                        },
                    ],
                    maxYScale // 최대 Y축 값 추가
                });
            } catch (error) {
                console.error("계산 중 오류 발생:", error);
                alert("계산 중 오류가 발생했습니다. 입력값을 조정해보세요.");
            } finally {
                setCalculating(false);
            }
        }, 100);
    }, [probPercent, cost, batchSize, targetWinCount, isMultipleWin, p]);

    // 페이지 로드 시 초기 계산 수행
    useEffect(() => {
        handleCalculate();
    }, []); // 컴포넌트 마운트 시 한 번만 실행

    // 개선된 차트 옵션 - x축 큰 숫자 표시 문제 해결
    const chartOptions = useMemo(() => ({
        responsive: true,
        maintainAspectRatio: false,
        animation: {
            duration: 500 // 애니메이션 시간 단축
        },
        scales: {
            y: {
                min: 0,
                max: chartInfo?.maxYScale || 100, // 적응형 Y축 스케일링
                title: { display: true, text: "확률 (%)" },
                ticks: {
                    callback: (v) => `${v}%`,
                    precision: 6 // 소수점 정밀도 증가
                },
            },
            x: {
                min: 1,
                max: chartInfo?.dynamicMaxAttempts,
                title: { display: true, text: "시행 횟수" },
                // 중요! x축이 실제 값을 표시하도록 설정
                type: 'linear',
                position: 'bottom',
                ticks: {
                    // 틱 간격을 사용자 정의
                    callback: function(value) {
                        // 값이 너무 크면 지수 표기법으로 변환
                        if (value >= 1e6) {
                            return value.toExponential(1).replace(/e\+?/, 'e');
                        } else if (value >= 1e3) {
                            // 천 단위 이상이면 K, M 접미사 사용
                            return (value / 1e3).toFixed(0) + 'K';
                        }
                        return value;
                    },
                    // 큰 데이터셋에 대해 표시할 틱 수 조정
                    maxTicksLimit: 8,
                    // 이 옵션이 중요: 실제 데이터 값에 따라 자동 스케일링
                    source: 'auto',
                },
            },
        },
        plugins: {
            tooltip: {
                callbacks: {
                    title: (context) => {
                        const rawX = context[0]?.parsed?.x ?? context[0]?.label ?? 0;
                        const xValue = typeof rawX === 'number' ? rawX : Number(String(rawX).replace(/[^\\d.]/g, '')); // 숫자 추출
                        if (isNaN(xValue)) return "시행 횟수: (불명확)";

                        const formattedValue = formatNumber(xValue);
                        const countDesc = Number(batchSize) > 1
                            ? ` 세트 (총 ${formatNumber(xValue * Number(batchSize))}회)`
                            : '회';

                        return `시행 횟수: ${formattedValue}${countDesc}`;
                    },
                    label: (context) => {
                        const value = parseFloat(context.formattedValue);
                        if (value < 0.01) {
                            // 아주 작은 값은 4자리까지 표시하고 싶으면 이렇게 (선택사항)
                            return `확률: ${value.toFixed(4)}%`;
                        }
                        // 일반 값은 무조건 2자리로 고정
                        return `확률: ${value.toFixed(2)}%`;
                    }
                }
            },
            legend: {
                display: true,
                position: 'top',
            },
        }
    }), [batchSize, chartInfo?.maxYScale,chartInfo?.dynamicMaxAttempts]);

    return (
        <div className="app-container">
            <div className="graph-section">
                <h1>기댓값 계산기</h1>
                {data && (
                    <div className="chart-wrapper">
                        <Line data={data} options={chartOptions} />
                        {chartInfo && (
                            <div className="chart-info">
                                그래프는 {chartInfo.isMultipleWin
                                ? `목표 ${chartInfo.targetWinCount}회`
                                : `확률 ${chartInfo.probability}%`}에 최적화된 <br />
                                {formatNumber(chartInfo.dynamicMaxAttempts)} {chartInfo.batchSize > 1 ? '세트' : '회'}
                                {chartInfo.batchSize > 1
                                    ? ` (총 ${formatNumber(chartInfo.dynamicMaxAttempts * chartInfo.batchSize)}회 뽑기)`
                                    : ''}
                                까지 표시됩니다.
                            </div>

                        )}
                    </div>
                )}
            </div>

            <div className="right-panel">
                <div className="input-controls">
                    <div className="input-group">
                        <label htmlFor="prob-percent">당첨 확률(%)</label>
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
                        <label htmlFor="batch-size">한 번에 뽑는 개수</label>
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
                            목표 당첨 횟수 설정
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
                            <p><strong>한 번에 뽑는 개수:</strong> {stats.batchSize}개</p>
                            <p>
                                <strong>1개 이상 당첨 확률:</strong>{" "}
                                {stats.batchSize === 1
                                    ? `${stats.pPercent}%`
                                    : `${Number(
                                        ((1 - Math.pow(1 - stats.pPercent / 100, stats.batchSize)) * 100).toFixed(10)
                                    )}%`}
                            </p>
                        </div>
                        <ul className="stats-list">
                            <li>
                                상위 20% <strong>
                                {formatNumber(stats.n20.n)} {stats.batchSize > 1 ? '세트' : '회'}{' '}
                                {stats.batchSize > 1 ? `(총 ${formatNumber(stats.totalTrials.n20)}회)` : ''}, 비용 {stats.n20.cost}
                            </strong>
                            </li>
                            <li>
                                평균 63.2% <strong>
                                {formatNumber(stats.n63.n)} {stats.batchSize > 1 ? '세트' : '회'}{' '}
                                {stats.batchSize > 1 ? `(총 ${formatNumber(stats.totalTrials.n63)}회)` : ''}, 비용 {stats.n63.cost}
                            </strong>
                            </li>
                            <li>
                                상위 80% <strong>
                                {formatNumber(stats.n80.n)} {stats.batchSize > 1 ? '세트' : '회'}{' '}
                                {stats.batchSize > 1 ? `(총 ${formatNumber(stats.totalTrials.n80)}회)` : ''}, 비용 {stats.n80.cost}
                            </strong>
                            </li>
                        </ul>

                    </div>
                )}
            </div>
        </div>
    );
}

export default App;