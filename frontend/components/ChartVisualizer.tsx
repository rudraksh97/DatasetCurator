import {
    Chart as ChartJS,
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    BarElement,
    Title,
    Tooltip,
    Legend,
    ArcElement,
} from 'chart.js';
import { Bar, Line, Pie, Scatter } from 'react-chartjs-2';

ChartJS.register(
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    BarElement,
    Title,
    Tooltip,
    Legend,
    ArcElement
);

interface ChartConfig {
    type: 'bar' | 'line' | 'scatter' | 'pie';
    title: string;
    data: any[];
    xField: string;
    yField?: string;
}

interface ChartVisualizerProps {
    config: ChartConfig;
}

export function ChartVisualizer({ config }: ChartVisualizerProps) {
    const { type, title, data, xField, yField } = config;

    // Prepare data for Chart.js
    const labels = data.map((row) => row[xField]);
    const values = data.map((row) => yField ? row[yField] : 0);

    // Random colors for pie chart
    const backgroundColors = type === 'pie'
        ? labels.map(() => `hsl(${Math.random() * 360}, 70%, 50%)`)
        : 'rgba(53, 162, 235, 0.5)';

    const chartData = {
        labels,
        datasets: [
            {
                label: yField || 'Count',
                data: values,
                backgroundColor: backgroundColors,
                borderColor: type === 'line' ? 'rgb(53, 162, 235)' : undefined,
                borderWidth: 1,
            },
        ],
    };

    const options = {
        responsive: true,
        maintainAspectRatio: false,
        layout: {
            padding: {
                bottom: 20
            }
        },
        plugins: {
            legend: {
                position: 'top' as const,
            },
            title: {
                display: true,
                text: title,
            },
        },
    };

    return (
        <div className="p-4 bg-white rounded-lg shadow-sm border border-gray-200 mt-4">
            <div className="h-[500px] w-full relative">
                {type === 'bar' && <Bar options={options} data={chartData} />}
                {type === 'line' && <Line options={options} data={chartData} />}
                {type === 'pie' && <Pie options={options} data={chartData} />}
                {type === 'scatter' && <Scatter options={options} data={chartData} />}
            </div>
            <div className="mt-2 text-xs text-gray-500 text-center">
                Config: {xField} vs {yField || 'Count'} (Type: {type})
            </div>
        </div>
    );
}
