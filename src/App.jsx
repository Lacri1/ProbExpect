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
    Filler
} from "chart.js";
import "./App.css";

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend, Filler);

// 로그팩토리얼 캐시
const LOG_FACTORIAL_CACHE = [0];
const logFactorial = (() => {
    const cache = [0];
    return (n) => {
        if (cache[n] != null) return cache[n];
        for (let i = cache.length; i <= n; i++) {
            cache[i] = cache[i - 1] + Math.log(i);
        }
        return cache[n];
    };
})();

const logCombination = (n, k) => {
    if (k < 0 || k > n) return -Infinity;
    return logFactorial(n) - logFactorial(k) - logFactorial(n - k);
};


const logBinomialProbability = (n, k, p) => {
    if (k < 0 || k > n) return -Infinity; // 유효하지 않은 k 값 처리
    if (p === 0) return k === 0 ? 0 : -Infinity;
    if (p === 1) return k === n ? 0 : -Infinity;
    // p가 0 또는 1에 매우 가까울 때 Math.log(0) 또는 Math.log(1) 문제를 피하기 위해 조건 추가
    const logP = (p > 1e-15) ? Math.log(p) : -Infinity;
    const logOneMinusP = (1 - p > 1e-15 && p < 0.9999999999) ? Math.log(1 - p) : -Infinity;

    const logComb = logCombination(n, k);
    // k*logP 계산 시 k=0이고 logP가 -Infinity인 경우 0으로 처리
    const kLogP = k > 0 && logP !== -Infinity ? k * logP : 0;
    // (n-k)*logOneMinusP 계산 시 n-k=0이고 logOneMinusP가 -Infinity인 경우 0으로 처리
    const nMinusKLogOneMinusP = (n - k > 0 && logOneMinusP !== -Infinity) ? (n - k) * logOneMinusP : 0;

    const result = logComb + kLogP + nMinusKLogOneMinusP;
    return result;
};

const binomialProbability = (n, k, p) => {
    if (k < 0 || k > n) return 0;
    if (p === 0) return k === 0 ? 1 : 0;
    if (p === 1) return k === n ? 1 : 0;
    if (p < 1e-15 && k > 0) return 0; // 확률이 극히 낮고 성공 횟수가 0이 아니면 0
    if (p > 0.9999999999 && k < n) return 0; // 확률이 극히 높고 실패 횟수가 0이 아니면 0

    let logProb; // logProb를 여기서 한 번만 선언합니다.

    // n이 크거나 p가 극단적이면 로그 확률 사용
    if (n >= 100 || p < 0.01 || p > 0.99) { // 큰 n 또는 극단적인 p 값일 때
        logProb = logBinomialProbability(n, k, p); // 값만 할당합니다.
        if (logProb === -Infinity || isNaN(logProb) || logProb < -700) return 0;
        const result = Math.exp(logProb);
        return isNaN(result) ? 0 : result;
    }


    // n이 작으면 직접 계산 시도
    try {
        const comb = Math.exp(logCombination(n, k)); // logCombination 사용
        const result = comb * Math.pow(p, k) * Math.pow(1 - p, n - k);
        return isNaN(result) ? 0 : result;
    } catch (e) {
        console.error('binomialProbability direct calculation error:', e);
        // 오류 발생 시 로그 확률로 대체 시도
        try {
            logProb = logBinomialProbability(n, k, p); // 값만 할당합니다.
            if (logProb === -Infinity || isNaN(logProb) || logProb < -700) return 0;
            const result = Math.exp(logProb);
            return isNaN(result) ? 0 : result;
        } catch (e2) {
            console.error('binomialProbability fallback log calculation error:', e2);
            return 0;
        }
    }
};


// 단일 당첨 확률 계산 함수 (천장 고려)
// 이 함수는 총 개별 시도 횟수를 인자로 받도록 수정되었습니다.
const calculateProbabilityWithPity = (totalIndividualAttempts, p, pityCount) => {
    if (totalIndividualAttempts <= 0 || p <= 0) return 0.0;

    // 천장 시스템이 없거나 비활성화된 경우 일반 확률 계산
    if (!pityCount || pityCount <= 0) {
        // 천장 없으면 일반 단일 당첨 확률 (총 시도 횟수 기준)
        return 1 - Math.pow(1 - p, totalIndividualAttempts);
    }

    // 총 시도 횟수가 천장 횟수 이상이면 무조건 1번 이상 당첨
    if (totalIndividualAttempts >= pityCount) {
        return 1.0;
    } else {
        // 총 시도 횟수가 천장 횟수 미만이면 천장 효과 없음. 일반 확률 계산.
        return 1 - Math.pow(1 - p, totalIndividualAttempts);
    }
};


// 수정된 천장 시스템을 고려한 확률 계산 함수 (복수 당첨)
// async 함수로 변경하고 DP 루프 내에 비동기 대기를 추가합니다.
const calculateProbabilityForMultipleWinsWithPity = (numBatches, p, batchSize, targetWinCount, pityCount) => {
    // 천장이 없거나 비활성화된 경우 또는 목표 당첨 횟수가 0 이하인 경우 일반 함수 또는 결과 반환
    if (!pityCount || pityCount <= 0) {
        const totalIndividualAttempts = numBatches * batchSize;
        let cumulativeProb = 0.0;
        // 이항 분포 합 계산은 계산량이 클 수 있으므로 비동기 대기 추가
        // 이 루프는 목표 당첨만 켰을 때 사용됩니다.
        for (let k = targetWinCount; k <= totalIndividualAttempts; k++) {
             // binomialProbability 함수 자체는 빠르지만, 루프 횟수가 많을 수 있습니다.
             cumulativeProb += binomialProbability(totalIndividualAttempts, k, p);
             // 비동기 대기 (예: 1000번 계산마다 yield)
             // 이 부분의 비동기 대기는 calculateProbabilityForMultipleWinsWithPity 함수 자체의 성능에 영향을 줍니다.
             // handleCalculate에서 루프를 돌 때 await를 거는 것으로 충분할 수 있습니다.
             // 여기서는 제거하여 함수 자체는 연속적으로 계산하도록 합니다. (필요 시 다시 추가)
             // if (k % 1000 === 0) {
             //     await new Promise(resolve => setTimeout(resolve, 0));
             // }
        }
        return cumulativeProb;
    }

    // 복수 당첨 및 일반 천장 사용 시 DP 로직
    // 이 로직은 목표 당첨 + 일반 천장 모드에서 사용됩니다.
    const totalAttempts = numBatches * batchSize; // 총 개별 시도 횟수
    const K = targetWinCount; // 목표 당첨 횟수
    const M = pityCount; // 천장 횟수 (pityCount는 1 이상이라고 가정)

    // 목표 당첨 횟수가 0 이하면 항상 확률 1.0
    if (K <= 0) return 1.0;

    // DP 테이블 초기화: dp[j][s] = '총 j번의 당첨'과 '최근 s번 연속 실패' 상태에 도달할 확률
    // j는 0부터 K-1까지, s는 0부터 M-1까지 (M번째 시도에서 천장 발동이므로 M-1 연속 실패 상태까지만 존재)
    // DP 테이블 크기가 (K) * (M) 이 됩니다. K나 M이 너무 크면 메모리 문제가 발생할 수 있습니다.
    let dp = Array(K).fill(0).map(() => Array(M).fill(0));
    let cumulativeProbAtLeastK = 0.0; // K번 이상 당첨될 누적 확률

    // 초기 상태: 0번 당첨, 0번 연속 실패 확률 1.0
    dp[0][0] = 1.0;

    // 개별 시도(attempt)를 totalAttempts 횟수만큼 시뮬레이션
    for (let i = 0; i < totalAttempts; i++) {
        const next_dp = Array(K).fill(0).map(() => Array(M).fill(0));
        let next_cumulativeProbAtLeastK = cumulativeProbAtLeastK;

        // 이전 상태 (j번 당첨, s번 연속 실패)를 순회
        for (let j = 0; j < K; j++) {
            for (let s = 0; s < M; s++) {
                if (dp[j][s] > 0) { // 해당 상태에 도달할 확률이 0보다 큰 경우만 고려

                    // --- 다음 시도 결과에 따른 상태 전이 계산 ---

                    // 1. 성공하는 경우 (일반 확률 p)
                    // 일반 확률로 당첨되면 당첨 횟수 +1, 연속 실패 횟수 0으로 리셋
                    if (p > 0) {
                        const probThisPath = dp[j][s] * p;
                        if (probThisPath > 1e-18) { // 매우 작은 확률은 무시하여 부동 소수점 오류 방지 및 성능 개선
                             if (j + 1 < K) { // 목표 횟수 K 미만이면 다음 DP 상태로 전이
                                 next_dp[j + 1][0] += probThisPath;
                             } else { // 목표 횟수 K 이상이면 누적 확률에 합산
                                 next_cumulativeProbAtLeastK += probThisPath;
                             }
                        }
                    }


                    // 2. 실패하는 경우 (확률 1-p)
                    // 실패하면 연속 실패 횟수 +1. 단, M-1에서 실패하면 M번째 시도에서 천장이므로 실패 상태로 전이되지 않음.
                    if (s + 1 < M) {
                         const probThisPath = dp[j][s] * (1 - p);
                         if (probThisPath > 1e-18) { // 매우 작은 확률 무시
                             next_dp[j][s + 1] += probThisPath;
                         }
                    }

                    // 3. 천장 발동하는 경우 (s + 1 === M 일 때)
                    // 현재 시도가 M번째 시도이고 이전까지 M-1번 연속 실패한 상태에서, 이번 시도 결과와 관계없이(확률 1.0으로) 당첨 횟수 +1, 연속 실패 횟수 0으로 리셋.
                    // 이는 "확률 당첨과 별개로 +1이 되는 구조"를 DP에 반영하는 핵심 부분입니다.
                    if (s + 1 === M) {
                        const probThisPath = dp[j][s]; // 이 상태에 도달할 확률 전부가 천장 발동으로 이어짐
                         if (probThisPath > 1e-18) { // 매우 작은 확률 무시
                             if (j + 1 < K) { // 목표 횟수 K 미만이면 다음 DP 상태로 전이
                                 next_dp[j + 1][0] += probThisPath;
                             } else { // 목표 횟수 K 이상이면 누적 확률에 합산
                                 next_cumulativeProbAtLeastK += probThisPath;
                             }
                         }
                    }
                     // 참고: 일반 성공(1)과 천장 발동(3)은 동일한 상태(next_dp[j+1][0])로 전이될 수 있으며, 실패(2)는 다른 상태로 전이됩니다.

                }
            }
        }
        dp = next_dp;
        cumulativeProbAtLeastK = next_cumulativeProbAtLeastK;

        // DP 루프 내부에서 비동기 대기 추가 (계산량 분산)
        // 1000번의 개별 시도 시뮬레이션마다 yield (조정 가능)
        // 계산 복잡도에 따라 이 값을 조정하여 UI 반응성을 높일 수 있습니다.
        // 이 부분은 일반 계산의 DP 로직이므로 async/await를 사용하지 않습니다.
        // if (i % 500 === 0) { // yield 빈도를 500으로 조정
        //      await new Promise(resolve => setTimeout(resolve, 0)); // async/await 제거
        // }
    }
    return Math.min(1.0, cumulativeProbAtLeastK); // 누적 확률은 최대 1.0
};

// 몬테카를로 시뮬레이션 기반 확률 계산 함수 (복수 당첨 + 천장)
// isCumulativePity 인자 추가 및 로직 분기
const simulateProbabilityWithPityAndMultipleWins = (
    maxBatches, // 시뮬레이션할 최대 배치 수 (그래프 x축 범위)
    p, // 개별 당첨 확률
    batchSize, // 한 번에 뽑는 개수
    targetWinCount, // 목표 당첨 횟수
    pityCount, // 천장 횟수
    numSimulations, // 실행할 시뮬레이션 횟수
    isCumulativePity // 누적 마일리지 모드 여부
) => {
    // 각 시뮬레이션에서 목표를 달성한 첫 번째 배치 번호를 기록 (목표 미달성 시 maxBatches + 1)
    const achievementBatches = Array(numSimulations).fill(maxBatches + 1);

    // 각 시뮬레이션 실행
    for (let sim = 0; sim < numSimulations; sim++) {
        let currentWins = 0; // 현재 시뮬레이션에서의 당첨 횟수
        let attemptsSinceLastWin = 0; // 마지막 당첨 후 시도 횟수 (일반 천장)
        let cumulativePityAttempts = 0; // 누적 마일리지 시도 횟수 (누적 마일리지 천장)

        // 최대 배치 수까지 시뮬레이션 진행
        for (let batchNum = 1; batchNum <= maxBatches; batchNum++) {
            // 현재 배치 내에서 개별 시도 시뮬레이션
            for (let i = 0; i < batchSize; i++) {
                attemptsSinceLastWin++;
                cumulativePityAttempts++; // 누적 마일리지 시도 횟수 증가

                let isWin = false;

                if (isCumulativePity) {
                    // 확률 당첨
                    if (Math.random() < p) {
                        isWin = true;
                    }
                    // 천장 당첨 (확률 당첨과 별개로 추가)
                    if (pityCount > 0 && cumulativePityAttempts > 0 && cumulativePityAttempts % pityCount === 0) {
                        currentWins++; // 확률 당첨과 별개로 1회 추가
                        // cumulativePityAttempts는 리셋되지 않음
                    }
                } else {
                    // 일반 천장 모드
                    // 일반 당첨 또는 천장 당첨 (천장 횟수에 도달 시)
                    if (Math.random() < p || (pityCount > 0 && attemptsSinceLastWin >= pityCount)) {
                         isWin = true;
                         attemptsSinceLastWin = 0; // 일반 천장은 당첨 시 카운트 리셋
                    }
                }

                if (isWin) {
                    currentWins++;
                }

                // 목표 당첨 횟수 달성 시 시뮬레이션 중단 및 달성 배치 기록
                if (currentWins >= targetWinCount) {
                    achievementBatches[sim] = batchNum; // 목표를 달성한 첫 번째 배치 번호 기록
                    break; // 현재 시뮬레이션 중단 (inner loop)
                }
            }
             // 목표 달성했으면 현재 시뮬레이션 중단 (outer loop)
             if(currentWins >= targetWinCount) break;
        }
    }

    // 모든 시뮬레이션이 끝난 후, 각 배치 시점에서의 목표 달성 누적 확률 계산 및 통계치 도출
    const labels = [];
    const probData = [];

    // 첫 번째 포인트 (0 시도 시 확률) 추가
    labels.push(0);
    probData.push(0);

    // 그래프 데이터 포인트 간격 설정 (샘플링)
    const maxDataPoints = 300; // 그래프 데이터 포인트 수
    const interval = Math.max(1, Math.floor(maxBatches / maxDataPoints));

    for(let batchNum = 1; batchNum <= maxBatches; batchNum += interval){
        // 해당 배치 번호까지 목표를 달성한 시뮬레이션 수 계산
        let achievedCount = 0;
        for(let sim = 0; sim < numSimulations; sim++){
            if(achievementBatches[sim] <= batchNum){
                achievedCount++;
            }
        }
        const prob = achievedCount / numSimulations; // 확률 계산
        labels.push(batchNum);
        probData.push((prob * 100).toFixed(4)); // 퍼센트로 변환 및 정밀도 설정
    }
    // 마지막 포인트 추가 (최대값)
    if(maxBatches > 0 && labels[labels.length - 1] !== maxBatches) {
        let prob;
        for(let sim = 0; sim < numSimulations; sim++){
            if(achievementBatches[sim] <= maxBatches){
                prob = probData[sim] / 100;
            }
        }
        prob = probData[numSimulations - 1] / 100;
        labels.push(maxBatches);
        probData.push((prob * 100).toFixed(4));
    }

    // 통계치 계산 (이분 탐색) - 반복 완료 후에 수행
    const sortedAchievementBatches = [...achievementBatches].sort((a, b) => a - b);

    const getPercentileBatch = (percentile) => {
        if (numSimulations === 0) return maxBatches + 1; // 시뮬레이션 없으면 최대값 반환
        const index = Math.min(numSimulations - 1, Math.floor(numSimulations * percentile / 100));
        return sortedAchievementBatches[index];
    };

    const batches20 = getPercentileBatch(20); // 상위 20% (하위 80% 지점)
    const batches63 = getPercentileBatch(63.2); // 평균 (지수 분포의 평균) 근사치
    const batches80 = getPercentileBatch(80); // 상위 80% (하위 20% 지점)

    return { labels, probData, batches20, batches63, batches80 };
};

// 이분 탐색을 사용하여 목표 확률에 도달하는 데 필요한 최소 시도 횟수 (배치 단위)를 찾는 함수 (천장 고려)
const findAttemptsForProbOptimizedWithPity = (
    targetProb, // 목표 누적 확률 (예: 0.8)
    p, // 개별 당첨 확률 (소수점)
    batchSize, // 한 번에 뽑는 개수
    isMultipleWin, // 복수 당첨 모드 여부
    targetWinCount, // 목표 당첨 횟수 (복수 당첨 모드일 때만 사용)
    maxBatchesLimit, // 이분 탐색 상한 (배치 단위)
    isPityEnabled, // 천장 시스템 사용 여부
    pityCount, // 천장 횟수
    isCumulativePity // 누적 마일리지 모드 여부
) => {
    if (targetProb <= 0) return 0; // 목표 확률 0 이하면 0회 시도
    if (targetProb >= 1) {
        // 목표 확률 100% 도달 시점 찾기 - 이분 탐색 필요
        // 하지만 무한 루프 방지를 위해 상한 제한 필요
        // 여기서는 근사적으로 상한 내에서 찾음
        // 실제 100% 도달 가능 여부는 calculateProbability 함수가 판단
    }
    if (p <= 0 && targetProb > 0) return maxBatchesLimit; // 확률 0%이면 목표 확률 달성 불가 (상한 반환)
    if (p >= 1 && isMultipleWin && targetWinCount > batchSize * maxBatchesLimit) return maxBatchesLimit; // 확률 100%라도 목표가 너무 높으면 상한 반환
    if (p >= 1 && !isMultipleWin && 1 > batchSize * maxBatchesLimit) return maxBatchesLimit; // 확률 100% 단일 당첨도 배치크기*상한 이내에 1회 당첨 안되면 상한 반환 (실제로는 1배치면 달성)


    let low = 1; // 최소 시도 횟수 (1 배치)
    let high = maxBatchesLimit; // 최대 시도 횟수 상한 (입력받은 제한값)
    let result = maxBatchesLimit; // 목표 확률 달성 시도 횟수 (기본값은 상한)

    while (low <= high) {
        const mid = Math.floor((low + high) / 2);
        let currentProb;

        // 천장 활성화 여부에 따라 적절한 확률 계산 함수 호출
        if (isMultipleWin) {
            currentProb = calculateProbabilityForMultipleWinsWithPity(
                mid, p, batchSize, targetWinCount, isPityEnabled ? pityCount : 0 // isPityEnabled 상태 전달 및 pityCount 사용
            );
        } else {
            // calculateProbabilityWithPity는 총 개별 시도 횟수와 개별 확률, 천장 횟수를 받음. batchSize는 필요 없음.
            currentProb = calculateProbabilityWithPity(
                mid * batchSize, p, isPityEnabled ? pityCount : 0 // isPityEnabled 상태 전달 및 pityCount 사용
            );
        }


        // 목표 확률과 비교
        if (currentProb >= targetProb) {
            result = mid; // 목표 확률 달성, 더 적은 시도 횟수 탐색
            high = mid - 1;
        } else {
            low = mid + 1; // 목표 확률 미달성, 더 많은 시도 횟수 탐색
        }
    }

    // 계산된 결과가 0이면 최소 1 배치로 설정 (0회 시도에 확률 0%가 아닌 경우 대비)
    return Math.max(1, result); // 최소 1 배치 반환
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


// 숫자를 적절한 형식으로 포맷팅하는 함수
const formatNumber = (num) => {
    if (typeof num !== 'number' || isNaN(num) || !isFinite(num)) return "N/A"; // 유효하지 않거나 무한대 처리

    // 100만 이상은 지수 표기법 사용
    if (num >= 1e6) {
        return num.toExponential(2).replace(/e\+?/, ' × 10^');
    }
    // 천 이상 100만 미만은 K 접미사 사용
    if (num >= 1000) {
        // 소수점 첫째 자리까지 표시, .0으로 끝나면 .0 제거
        const formatted = (num / 1000).toFixed(1);
        return formatted.endsWith('.0') ? formatted.slice(0, -2) + 'K' : formatted + 'K';
    }
    // 1보다 작은 양수는 소수점 여러 자리 표시
    if (num > 0 && num < 1) {
        // 적절한 정밀도로 표시 (예: 최대 6자리)
        return num.toFixed(Math.max(2, -Math.floor(Math.log10(num)) + 2)); // 로그 스케일로 정밀도 결정
    }


    return num.toLocaleString('en-US', { maximumFractionDigits: 2 }); // 기본값: 소수점 둘째 자리까지
};


function App() {
    const [probPercent, setProbPercent] = useState("0.6");  // 기본값
    const [cost, setCost] = useState("1000");
    const [batchSize, setBatchSize] = useState("10");
    const [data, setData] = useState(null);
    const [stats, setStats] = useState(null);
    const [dynamicMaxAttempts, setDynamicMaxAttempts] = useState(180);  // 기본 최대 시도 횟수 (예시 값)
    const [isMultipleWin, setIsMultipleWin] = useState(false); // 예시: 기본값 false
    const [targetWinCount, setTargetWinCount] = useState("10"); // 예시: 기본값 10
    const [calculating, setCalculating] = useState(false);
    const [chartInfo, setChartInfo] = useState(null); // 차트 정보 상태 추가
    // 천장 시스템 관련 상태 추가
    const [isPityEnabled, setIsPityEnabled] = useState(false); // 예시: 기본값 false
    const [pityCount, setPityCount] = useState("300"); // 예시: 기본값 300
    const [isCumulativePity, setIsCumulativePity] = useState(false); // 누적 마일리지 모드 상태 추가

    // 입력된 확률을 소수점으로 변환
    const p = useMemo(() => {
        // 입력값을 항상 비율로 처리
        const percent = Number(probPercent);
        return isNaN(percent) || percent < 0 ? 0 : percent / 100; // 유효하지 않은 값 처리
    }, [probPercent]);

    // 계산이 무거울 때 사용할 제한값
    const MAX_SAFE_COMPUTATION = 100000;  // 최대 계산 범위 증가 (테스트를 위해 100k으로 낮춤)
    const COMPUTATION_WARNING_THRESHOLD = 10000; // 계산량 경고 임계값

    // 안전한 계산 체크 (대략적인 휴리스틱) - 캡처된 값을 사용하도록 수정
    const isHeavyComputation = useCallback((currentIsMultipleWin, currentTargetWinCount, currentP, currentBatchSize) => {
        // 복수 당첨 모드이고, 목표 당첨 횟수가 높고, 개별 확률이 매우 낮을 때
        // 천장 시스템 사용 시 DP는 totalAttempts * K * M 에 비례
        // 천장 미사용 시 findAttemptsForProbOptimized는 이분 탐색 O(log N)
        // DP 계산 복잡성을 고려하여 임계값 조정 필요
        if (currentIsMultipleWin && currentP > 0) {
            // 대략적인 DP 상태 공간 크기: numBatches * K * M
            // numBatches의 상한은 findAttemptsForProbOptimized에서 결정됨
            // 여기서는 계산 전 예측이므로, 대략적인 totalAttempts 상한을 MAX_SAFE_COMPUTATION * batchSize로 가정
            const estimatedTotalAttempts = MAX_SAFE_COMPUTATION * currentBatchSize;
            const estimatedDPStates = estimatedTotalAttempts * currentTargetWinCount * Number(pityCount); // pityCount 상태 직접 사용

            // 임계값은 임의로 설정, 조정 필요
            const DP_COMPUTATION_THRESHOLD = 50000000; // 예: 5천만 상태 초과 시 경고
            if (isPityEnabled && Number(pityCount) > 0) { // 천장 사용 시
                return estimatedDPStates > DP_COMPUTATION_THRESHOLD;
            } else { // 천장 미사용 시 (이분 탐색)
                // 복잡성 판단 로직 수정 필요
                // 복수 당첨 & 천장 미사용 시, 이항 분포 합 계산 복잡성
                // N = totalAttempts, K = targetWinCount
                // 복잡성은 sum(C(N, i)) for i=K to N 이므로 N과 N-K 중 작은 값에 따라 다름. O(N) 또는 O(N-K)
                // N이 MAX_SAFE_COMPUTATION * batchSize 가 될 수 있으므로 이 경우에도 계산량이 클 수 있음.
                // 대략적인 계산량: MAX_SAFE_COMPUTATION * batchSize - targetWinCount
                const estimatedBinomialCalcs = MAX_SAFE_COMPUTATION * currentBatchSize - currentTargetWinCount;
                const BINOMIAL_COMPUTATION_THRESHOLD = 100000; // 예: 100만 계산 초과 시 경고
                return estimatedBinomialCalcs > BINOMIAL_COMPUTATION_THRESHOLD;

            }
        }


        return false; // 복수 당첨 아니거나 확률 0이면 제외
    }, [isPityEnabled, pityCount]); // isPityEnabled와 pityCount를 종속성에 추가


    // 계산 함수
    const handleCalculate = useCallback(() => {
        setCalculating(true);

        // 계산 시작 시점의 모든 상태 값을 캡처
        const currentProbPercent = probPercent;
        const currentCost = cost;
        const currentBatchSize = batchSize;
        const currentTargetWinCount = targetWinCount;
        const currentIsMultipleWin = isMultipleWin;
        const currentIsPityEnabled = isPityEnabled;
        const currentPityCount = pityCount;
        const currentIsCumulativePity = isCumulativePity;

        // 문자열 상태를 숫자로 변환
        const inputProb = Number(currentProbPercent);
        const c = Number(currentCost);
        const batch = Number(currentBatchSize);
        const target = Number(currentTargetWinCount);
        const pity = currentIsPityEnabled ? Number(currentPityCount) : 0;

        // 유효성 검사
        if (
            isNaN(inputProb) || inputProb < 0 ||
            isNaN(c) || c <= 0 ||
            isNaN(batch) || batch <= 0 ||
            (currentIsMultipleWin && (isNaN(target) || target < 1)) ||
            (currentIsPityEnabled && (isNaN(pity) || pity < 1))
        ) {
            alert("모든 입력란에 올바른 숫자를 입력해주세요 (확률 0% 및 목표/천장 횟수 0 제외). 미국식 숫자 표기법(예: 0.5)을 사용해주세요.");
            setCalculating(false); // 유효성 검사 실패 시 calculating 상태 해제
            return;
        }

        // 특정 조건 (확률 낮음 + 목표 당첨 횟수 + 천장 시스템)에서 계산 차단
        if (inputProb < 0.1 && currentIsMultipleWin && currentIsPityEnabled && target > 0 && pity > 0) {
             alert("확률이 0.1 미만에서서 목표 당첨 횟수 및 천장 시스템이 함께 활성화된 경우 계산을 제공하지 않습니다.");
             setCalculating(false);
             setData(null);
             setStats(null);
             setChartInfo(null);
             return;
        }

        // 확률 0%인데 목표 당첨 횟수가 0보다 크거나, 확률 100%인데 목표 당첨 횟수가 총 시도 가능 횟수보다 클 경우 특수 처리
        if (inputProb === 0) { // 확률 0%인 경우
            if (currentIsMultipleWin && target > 0) {
                alert("당첨 확률이 0%이므로 목표 당첨 횟수를 달성할 수 없습니다.");
                setCalculating(false);
                setData(null);
                setStats(null);
                setChartInfo(null);
                return;
            }
            if (!currentIsMultipleWin || target === 0) {
                // 확률 0%이고 목표 당첨 횟수가 0이면 항상 달성 가능 -> 계산 진행은 하되 결과는 0%
            }
        }

        if (inputProb >= 100) {
            const totalPossibleAtMaxAttempts = MAX_SAFE_COMPUTATION * batch;
            if (currentIsMultipleWin && target > totalPossibleAtMaxAttempts) {
                alert(`당첨 확률 100%에도 최대 시도 횟수(${formatNumber(MAX_SAFE_COMPUTATION)} 세트) 내에서 목표 당첨 횟수(${formatNumber(target)}회)를 달성할 수 없습니다.`);
                setCalculating(false);
                setData(null);
                setStats(null);
                setChartInfo(null);
                return;
            }
        }

        // 복수 당첨 + 마일리지 방식 시뮬레이션 사용 여부
        const useSimulationMode = currentIsMultipleWin && currentIsPityEnabled && pity > 0;

        // 일반 계산 (비 시뮬레이션) 중 복수 당첨 + 일반 천장 DP 계산 여부
        const useDPForGraph = currentIsMultipleWin && currentIsPityEnabled && pity > 0 && !currentIsCumulativePity;

        // 실제 계산 로직 (async 함수로 선언)
        const performCalculation = async () => {
            try {
                const currentP = Number(currentProbPercent) / 100;

                // 그래프 데이터 생성 및 통계치 계산 변수 초기화
                let labels = [];
                let probData = [];
                let batches20 = MAX_SAFE_COMPUTATION;
                let batches63 = MAX_SAFE_COMPUTATION;
                let batches80 = MAX_SAFE_COMPUTATION;
                let maxProbForScaling = 0;

                // 최적 시도 횟수 계산 (배치 단위) - 그래프 X축 범위 결정의 기초
                let estimatedMaxBatchesFromProb = findAttemptsForProbOptimizedWithPity(
                    0.995, currentP, batch, currentIsMultipleWin, target,
                    MAX_SAFE_COMPUTATION, currentIsPityEnabled, pity, currentIsCumulativePity
                );

                let finalMaxBatches = Math.max(1, estimatedMaxBatchesFromProb);

                // --- finalMaxBatches 계산 로직 수정 ---
                // 천장 시스템이 활성화된 경우, 최대 시도 횟수 상한을 천장 횟수와 목표 횟수에 기반하여 조정합니다.
                if (currentIsPityEnabled && pity > 0) {
                    const batchesPerPity = batch > 0 ? Math.ceil(pity / batch) : pity; // 천장 횟수를 배치 수로 환산
                    if (currentIsMultipleWin && target > 0) {
                         // 복수 당첨 + 천장 모드: 최소한 '목표 * 천장' 개별 시도에 해당하는 배치 수 이상 필요
                         const estimatedTotalAttemptsNeeded = target * pity;
                         const estimatedBatchesNeeded = batch > 0 ? Math.ceil(estimatedTotalAttemptsNeeded / batch) : estimatedTotalAttemptsNeeded;

                         // finalMaxBatches는 '99.5% 도달 시점', '목표*천장 관련 배치 수', '천장 배치*2' 중 가장 큰 값과 MAX_SAFE_COMPUTATION 중 작은 값
                         finalMaxBatches = Math.max(finalMaxBatches, estimatedBatchesNeeded, batchesPerPity * 2);

                    } else { // 단일 당첨 + 일반 천장: 천장 횟수 배치 이상이면 100% 도달
                         // finalMaxBatches는 '99.5% 도달 시점', '천장 배치*2' 중 큰 값과 MAX_SAFE_COMPUTATION 중 작은 값
                         finalMaxBatches = Math.max(finalMaxBatches, batchesPerPity * 2);
                    }
                    // 최종 상한 적용
                    finalMaxBatches = Math.min(finalMaxBatches, MAX_SAFE_COMPUTATION);
                     // 그래프 최소 범위 보장
                    finalMaxBatches = Math.max(finalMaxBatches, 5); // 최소 5 배치 보장

                } else { // 천장 시스템 미사용 시 (단일 당첨 또는 복수 당첨 + 천장 Off)
                    // 99.5% 도달 시점 기반 finalMaxBatches를 사용하되, MAX_SAFE_COMPUTATION으로 제한
                    finalMaxBatches = Math.min(finalMaxBatches, MAX_SAFE_COMPUTATION);
                     // 그래프 최소 범위 보장
                    finalMaxBatches = Math.max(finalMaxBatches, 5); // 최소 5 배치 보장
                }

                setDynamicMaxAttempts(finalMaxBatches);

                // 첫 번째 포인트 (0 시도 시 확률) 추가 (기존 로직 유지)
                let initialProb = (currentIsMultipleWin && target === 0) ? 1.0 : 0.0;
                labels.push(0);
                probData.push((initialProb * 100).toFixed(4));
                maxProbForScaling = Math.max(maxProbForScaling, initialProb);


                // --- 그래프 데이터 생성 및 통계 계산 --- //

                // 복수 당첨 & 천장 시스템 사용 시 몬테카를로 시뮬레이션 사용 (마일리지 On/Off 모두 포함)
                if (useSimulationMode) { // 시뮬레이션 (복수 당첨 + 천장 On - 마일리지 On/Off)
                    const numSimulations = 20000; // 시뮬레이션 횟수
                    console.log(`Starting Monte Carlo simulation (${currentIsCumulativePity ? 'Cumulative Pity' : 'Regular Pity'}) with ${numSimulations} runs for maxBatches: ${finalMaxBatches}`);

                    // 시뮬레이션 함수 호출 시 isCumulativePity 상태 전달
                    const simResult = simulateProbabilityWithPityAndMultipleWins(
                        finalMaxBatches, currentP, batch, target, pity, numSimulations,
                        currentIsCumulativePity
                    );

                    labels.push(...simResult.labels.slice(1));
                    probData.push(...simResult.probData.slice(1));

                     // 시뮬레이션 결과에서 통계치 배치 수 가져오기
                    batches20 = simResult.batches20;
                    batches63 = simResult.batches63;
                    batches80 = simResult.batches80;

                     // 시뮬레이션 결과에서 최대 확률 찾기
                    maxProbForScaling = simResult.probData.reduce((max, current) => Math.max(max, parseFloat(current) / 100), initialProb);

                } else { // 그 외의 경우 (단일 당첨 또는 복수 당첨 & 천장 미사용 또는 복수 당첨 + 일반 천장 DP) 기존 계산 방식 사용

                    const maxDataPoints = 300; // 그래프 데이터 포인트 수
                    const step = Math.max(1, Math.floor(finalMaxBatches / maxDataPoints));

                    // 그래프 데이터 포인트 샘플링 (배치 단위로) with yielding
                    for (let n = 1; n <= finalMaxBatches; n += step) {
                        let prob;
                        // calculateProbabilityForMultipleWinsWithPity 함수는 이제 async가 아님.
                        // 복수 당첨 + 일반 천장 (useDPForGraph가 true)일 때도 이 함수 사용.
                        if (currentIsMultipleWin) {
                             // 복수 당첨 & 천장 Off 또는 복수 당첨 & 일반 천장 DP
                            prob = calculateProbabilityForMultipleWinsWithPity(n, currentP, batch, target, currentIsPityEnabled && !currentIsCumulativePity ? pity : 0);
                        } else {
                            // 단일 당첨 (천장 On/Off는 함수 내에서 처리)
                            prob = calculateProbabilityWithPity(n * batch, currentP, pity);
                        }
                        if (isNaN(prob) || prob < 0) prob = 0;
                        if (prob > 1) prob = 1;
                        maxProbForScaling = Math.max(maxProbForScaling, prob);
                        labels.push(n);
                        probData.push((prob * 100).toFixed(4));

                         // 각 스텝마다 UI 업데이트 기회를 줌 (무한 로딩 방지)
                         // 일반 계산 시 이 yield가 성능 문제를 일으킬 수 있으므로 제거를 시도합니다.
                         // 필요한 경우 더 효율적인 비동기 처리 방법을 고려합니다.
                         // await new Promise(resolve => setTimeout(resolve, 0)); // yield 제거
                    }

                    // 마지막 포인트 추가 (최대값)
                    if (finalMaxBatches > 0 && labels[labels.length - 1] !== finalMaxBatches) {
                        let prob;
                        // calculateProbabilityForMultipleWinsWithPity 함수는 이제 async가 아님.
                        // 복수 당첨 + 일반 천장 (useDPForGraph가 true)일 때도 이 함수 사용.
                        if (currentIsMultipleWin) {
                             prob = calculateProbabilityForMultipleWinsWithPity(finalMaxBatches, currentP, batch, target, currentIsPityEnabled && !currentIsCumulativePity ? pity : 0);
                        } else {
                            prob = calculateProbabilityWithPity(finalMaxBatches * batch, currentP, pity);
                        }
                        if (isNaN(prob) || prob < 0) prob = 0;
                        if (prob > 1) prob = 1;

                        maxProbForScaling = Math.max(maxProbForScaling, prob);
                        labels.push(finalMaxBatches);
                        probData.push((prob * 100).toFixed(4));

                         // 마지막 포인트 추가 후에도 UI 업데이트 기회 제공
                         // await new Promise(resolve => setTimeout(resolve, 0)); // yield 제거
                    }


                    // 통계치 계산 (이분 탐색) - 반복 완료 후에 수행
                    // 통계 계산 시에는 최종 finalMaxBatches 범위 내에서 정확한 값을 찾도록 함.
                     batches20 = findAttemptsForProbOptimizedWithPity(
                         0.2, currentP, batch, currentIsMultipleWin, target, finalMaxBatches, // finalMaxBatches 사용
                         currentIsPityEnabled, pity, currentIsCumulativePity // isCumulativePity 전달
                     );
                     batches63 = findAttemptsForProbOptimizedWithPity(
                         0.6321, currentP, batch, currentIsMultipleWin, target, finalMaxBatches, // finalMaxBatches 사용
                         currentIsPityEnabled, pity, currentIsCumulativePity // isCumulativePity 전달
                     );
                     batches80 = findAttemptsForProbOptimizedWithPity(
                         0.8, currentP, batch, currentIsMultipleWin, target, finalMaxBatches, // finalMaxBatches 사용
                         currentIsPityEnabled, pity, currentIsCumulativePity // isCumulativePity 전달
                     );

                    // 계산된 배치 수가 finalMaxBatches를 초과하지 않도록 보정 (유지)
                    if (batches20 <= 0 || batches20 > finalMaxBatches) batches20 = finalMaxBatches;
                    if (batches63 <= 0 || batches63 > finalMaxBatches) batches63 = finalMaxBatches;
                    if (batches80 <= 0 || batches80 > finalMaxBatches) batches80 = finalMaxBatches;
                }

                // --- 계산 결과 상태 업데이트 --- //
                const maxYScale = maxProbForScaling * 100 < 10
                    ? Math.max(1, Math.ceil(maxProbForScaling * 100 * 1.5))
                    : 100;

                // 천장 시스템 정보를 포함한 차트 정보 생성 (마일리지 포함)
                const pityInfo = currentIsPityEnabled && pity > 0 ? {
                    isPityEnabled: true,
                    pityCount: pity,
                    isCumulativePity: currentIsCumulativePity,
                    batchesForPity: batch > 0 ? Math.ceil(pity / batch) : null,
                    // 복수 당첨 모드에서 최대 시도 횟수까지 천장으로 얻는 확정 당첨 횟수 계산 (시뮬레이션 모드에서만 의미)
                    confirmWinsAtMaxAttempts: useSimulationMode && currentIsMultipleWin && pity > 0 && batch > 0 && finalMaxBatches > 0
                        ? Math.floor((finalMaxBatches * batch) / pity)
                        : null
                } : { isPityEnabled: false };


                // 통계 설정 (마일리지 정보 포함)
                 // 시뮬레이션 또는 기존 계산 결과에서 얻은 batchesXX 값을 사용하여 통계 업데이트
                setStats({
                    n20: { n: batches20, cost: formatNumber(batches20 * c) },
                    n63: { n: batches63, cost: formatNumber(batches63 * c) },
                    n80: { n: batches80, cost: formatNumber(batches80 * c) },
                    pIndividual: currentP,
                    pPercent: ((str => str.replace(/\.?0+$/, ""))((currentP * 100).toFixed(currentP < 0.01 ? 6 : 2))),
                    batchSize: batch,
                    totalTrials: {
                        n20: batches20 * batch,
                        n63: batches63 * batch,
                        n80: batches80 * batch
                    },
                    pityEnabled: currentIsPityEnabled,
                    pityCount: pity,
                    isCumulativePity: currentIsCumulativePity
                });

                setChartInfo({
                    isMultipleWin: currentIsMultipleWin,
                    targetWinCount: target,
                    probability: (() => {
                        const raw = currentP * 100;
                        if (!isNaN(raw) && raw > 0) {
                            let fixed = raw < 0.001 ? raw.toFixed(6) : raw < 0.01 ? raw.toFixed(4) : raw.toFixed(2);
                            return fixed.replace(/\.?0+$/, "");
                        }
                        return "0";
                    })(),
                    dynamicMaxAttempts: finalMaxBatches,
                    batchSize: batch,
                    maxYScale,
                    pityInfo: pityInfo
                });

                setData({
                    labels,
                    datasets: [
                        {
                            label: currentIsMultipleWin
                                ? `${formatNumber(target)}회 이상 당첨 확률 (%)`
                                : "당첨 확률 (%)",
                            data: probData,
                            borderColor: "rgba(75,192,192,1)",
                            backgroundColor: "rgba(75,192,192,0.2)",
                            fill: true,
                            tension: 0.1,
                             // 시뮬레이션 모드일 때 점 크기 작게
                             pointRadius: 1, // 모든 계산 방식에서 점 크기 작게 고정
                             pointHoverRadius: 3, // 모든 계산 방식에서 마우스 오버 시 점 크기 작게 고정
                        },
                    ],
                    maxYScale
                });
            } catch (error) {
                console.error("계산 중 오류 발생:", error);
                alert("계산 중 오류가 발생했습니다. 입력값을 조정해보세요.");
            } finally {
                setCalculating(false);
            }
        };

        // setCalculating(true) 호출 후 UI 갱신 시간을 주고 비동기 실행
        // 모든 계산 시작 전 UI 업데이트를 위해 항상 setTimeout 사용
        setTimeout(performCalculation, 50); // setTimeout 딜레이 50ms (조정 가능)

    }, [probPercent, cost, batchSize, targetWinCount, isMultipleWin, isPityEnabled, pityCount, isCumulativePity]);

    // 페이지 로드 시 초기 계산 수행
    useEffect(() => {
        handleCalculate();
    }, []);

    // 개선된 차트 옵션 (마일리지 정보 툴팁 추가)
    const chartOptions = useMemo(() => ({
        responsive: true,
        maintainAspectRatio: false,
        animation: {
            duration: 500 // 애니메이션 시간 단축
        },
        scales: {
             y: {
                 min: 0,
                 max: chartInfo?.maxYScale || 100,
                 title: { display: true, text: "확률 (%)" },
                 ticks: {
                     callback: (v) => `${v}%`,
                     precision: 6
                 },
                 grid: {
                     color: 'rgba(200, 200, 200, 0.2)',
                 }
             },
             x: {
                 min: 0,
                 max: chartInfo?.dynamicMaxAttempts,
                 title: { display: true, text: "시행 횟수 (세트)" },
                 type: 'linear',
                 position: 'bottom',
                 ticks: {
                     callback: function(value) { return value.toString(); },
                 },
                 grid: {
                     color: 'rgba(200, 200, 200, 0.2)',
                 }
             },
         },
        plugins: {
            tooltip: {
                callbacks: {
                    title: (context) => {
                         const rawX = context[0]?.parsed?.x ?? context[0]?.label ?? 0;
                         const xValue = typeof rawX === 'number' && !isNaN(rawX) ? rawX : Number(String(rawX).replace(/[^\\d.]/g, '')) || 0;
                         if (isNaN(xValue)) return "시행 횟수: (불명확)";
                         const formattedBatches = formatNumber(xValue);
                         const currentBatchSize = Number(batchSize);
                         const totalAttempts = xValue * currentBatchSize;
                         const formattedAttempts = formatNumber(totalAttempts);
                         if (currentBatchSize > 1) {
                            return `시행 횟수: ${formattedBatches} 세트 (총 ${formattedAttempts}회)`;
                         } else {
                            return `시행 횟수: ${formattedAttempts}회`;
                         }
                    },
                    label: (context) => {
                        const value = parseFloat(context.formattedValue);
                        if (value > 0 && value < 0.01) {
                            return `확률: ${value.toFixed(4)}%`;
                        } else if (value >= 0.01) {
                            return `확률: ${value.toFixed(2)}%`;
                        }
                        return `확률: ${value}%`;
                    },
                }
            },
             legend: {
                 display: true,
                 position: 'top',
             },
         }
     }), [batchSize, chartInfo?.maxYScale, chartInfo?.dynamicMaxAttempts, chartInfo?.pityInfo]);

    return (
        <div className="app-container">
            <div className="graph-section">
                <h1>기댓값 계산기</h1>
                {data && (
                    <div className="chart-wrapper">
                        <Line data={data} options={chartOptions} />
                    </div>
                )}
                {chartInfo && (
                    <div className="chart-info">
                        그래프는 {chartInfo.isMultipleWin
                        ? `목표 ${formatNumber(chartInfo.targetWinCount)}회 당첨`
                        : `단일 당첨`}에 최적화된 <br />
                        {formatNumber(chartInfo.dynamicMaxAttempts)} {chartInfo.batchSize > 1 ? '세트' : '회'}
                        {chartInfo.batchSize > 1
                            ? ` (총 ${formatNumber(chartInfo.dynamicMaxAttempts * chartInfo.batchSize)}회 뽑기)`
                            : ''}
                        까지 표시됩니다.
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

                    <div className="input-group checkbox-group">
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

                    {/* 천장 시스템 토글 및 입력 추가 */}
                    <div className="input-group checkbox-group">
                        <label>
                            <input
                                type="checkbox"
                                checked={isPityEnabled}
                                onChange={() => setIsPityEnabled(!isPityEnabled)}
                                disabled={calculating}
                            />
                            천장 시스템 사용
                        </label>
                         {/* 천장 시스템 도움말 아이콘 및 텍스트 (마일리지 정보 포함) */}
                         {isPityEnabled && (
                             <span className="help-container">
                                 <span className="help-icon">?</span>
                                 <div className="input-help tooltip">
                                     {isCumulativePity ? ( // 마일리지 모드일 때
                                         `마일리지형: ${formatNumber(Number(pityCount))}회마다 당첨 횟수 +1`
                                     ) : ( // 일반 천장 모드일 때
                                         `일반형: 최근 ${formatNumber(Number(pityCount) - 1)}회 미당첨 시 ${formatNumber(Number(pityCount))}번째 확정 당첨`
                                     )}
                                     <br/>
                                     몬테카를로 시뮬레이션 방식으로 계산되며,<br/>
                                     시행 횟수가 적을 경우 오차가 발생할 수 있습니다.<br/>
                                     ※ 시뮬레이션 복잡도로 인해 계산 시간이 길어질 수 있습니다.
                                 </div>
                             </span>
                         )}
                    </div>

                    {isPityEnabled && (
                        <div className="input-group">
                            <label htmlFor="pity-count">천장 횟수</label>
                            <input
                                id="pity-count"
                                type="text"
                                value={pityCount}
                                onChange={(e) => setPityCount(e.target.value)}
                                className="input-number"
                                disabled={calculating}
                            />
                        </div>
                    )}

                    {/* 마일리지 방식 체크박스 추가 */}
                    {isPityEnabled && (
                        <div className="input-group checkbox-group">
                            <label>
                                <input
                                    type="checkbox"
                                    checked={isCumulativePity}
                                    onChange={() => setIsCumulativePity(!isCumulativePity)}
                                    disabled={calculating}
                                />
                                마일리지 방식
                            </label>
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
                            <p><strong>한 번에 뽑는 개수:</strong> {formatNumber(stats.batchSize)}개</p>
                            <p>
                                <strong>1개 이상 당첨 확률:</strong>{" "}
                                {stats.batchSize === 1
                                    ? `${stats.pPercent}%`
                                    : `${formatNumber(
                                          ((1 - Math.pow(1 - stats.pIndividual, stats.batchSize)) * 100)
                                      )}%`}
                            </p>
                            {stats.pityEnabled && stats.pityCount > 0 && (
                                 <p>
                                     <strong>천장:</strong>{" "}
                                     {stats.isCumulativePity ?
                                         ` ${formatNumber(stats.pityCount)}회 시도마다 확정 당첨` :
                                         <>
                                             {` ${formatNumber(stats.pityCount - 1)}회 미당첨 시 ${formatNumber(stats.pityCount)}번째 확정 당첨`}
                                         </>
                                     }
                                 </p>
                            )}
                        </div>
                        <ul className="stats-list">
                             {!isNaN(stats.n20.n) && isFinite(stats.n20.n) && (
                                 <li>
                                     상위 20% <strong>
                                     {formatNumber(stats.n20.n)} {stats.batchSize > 1 ? '세트' : '회'}{' '}
                                     {stats.batchSize > 1 ? `(총 ${formatNumber(stats.totalTrials.n20)}회)` : ''}, 비용 {stats.n20.cost}
                                 </strong>
                                 </li>
                             )}
                             {!isNaN(stats.n63.n) && isFinite(stats.n63.n) && (
                                 <li>
                                     평균 63.2%
                                      <span className="help-container stats-help-tooltip">
                                         <span className="help-icon average-icon"></span>
                                         <div className="input-help tooltip">
                                             평균 시도 횟수까지 당첨될 확률은 약 63.2%이다. <br/>
                                             확률 1%를 100번 뽑을 때, 1개 이상 당첨될 확률은 약 63.2%
                                         </div>
                                      </span>
                                    <strong>
                                     {formatNumber(stats.n63.n)} {stats.batchSize > 1 ? '세트' : '회'}{' '}
                                     {stats.batchSize > 1 ? `(총 ${formatNumber(stats.totalTrials.n63)}회)` : ''}, 비용 {stats.n63.cost}
                                 </strong>
                                 </li>
                             )}
                             {!isNaN(stats.n80.n) && isFinite(stats.n80.n) && (
                                 <li>
                                     상위 80% <strong>
                                     {formatNumber(stats.n80.n)} {stats.batchSize > 1 ? '세트' : '회'}{' '}
                                     {stats.batchSize > 1 ? `(총 ${formatNumber(stats.totalTrials.n80)}회)` : ''}, 비용 {stats.n80.cost}
                                 </strong>
                                 </li>
                             )}

                         </ul>
                    </div>
                )}

             </div>
         </div>
     );
}

export default App;