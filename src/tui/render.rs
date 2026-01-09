use ratatui::{
    prelude::*,
    widgets::*,
};

use crate::config::{
    BATCH_SIZE, EPS_END, EPS_START, EPS_DECAY, GAMMA, LR, MEMORY_CAPACITY, TAU, TOTAL_EPISODES,
};
use crate::tui::DashboardState;

pub fn ui(frame: &mut Frame, state: &DashboardState) {
    let main_layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3), // Title
            Constraint::Min(10),   // Main content
            Constraint::Length(3), // Footer
        ])
        .split(frame.area());

    // Title
    let title = Paragraph::new("DQN CartPole Training Dashboard")
        .style(
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        )
        .alignment(Alignment::Center)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Cyan)),
        );
    frame.render_widget(title, main_layout[0]);

    // Main content area
    let content_layout = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(70), // Charts
            Constraint::Percentage(30), // Stats
        ])
        .split(main_layout[1]);

    // Charts area
    let charts_layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage(60), // Episode duration chart
            Constraint::Percentage(40), // Epsilon chart
        ])
        .split(content_layout[0]);

    // Episode Duration Chart
    render_duration_chart(frame, charts_layout[0], state);

    // Epsilon Chart
    render_epsilon_chart(frame, charts_layout[1], state);

    // Stats Panel
    render_stats_panel(frame, content_layout[1], state);

    // Footer
    let status = if state.training_complete {
        "Training Complete! Press 'q' to exit"
    } else {
        "Training in progress... Press 'q' to quit"
    };
    let footer = Paragraph::new(status)
        .style(Style::default().fg(Color::Yellow))
        .alignment(Alignment::Center)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::DarkGray)),
        );
    frame.render_widget(footer, main_layout[2]);
}

fn render_duration_chart(frame: &mut Frame, area: Rect, state: &DashboardState) {
    let data: Vec<(f64, f64)> = state.episode_durations.clone();

    // Calculate moving average for smoother visualization
    let moving_avg: Vec<(f64, f64)> = state
        .episode_durations
        .iter()
        .enumerate()
        .map(|(i, (ep, _))| {
            let start = if i >= 20 { i - 20 } else { 0 };
            let window: Vec<f64> = state.episode_durations[start..=i]
                .iter()
                .map(|(_, d)| *d)
                .collect();
            let avg = window.iter().sum::<f64>() / window.len() as f64;
            (*ep, avg)
        })
        .collect();

    let datasets = vec![
        Dataset::default()
            .name("Episode Duration")
            .marker(symbols::Marker::Braille)
            .style(Style::default().fg(Color::Cyan))
            .graph_type(GraphType::Scatter)
            .data(&data),
        Dataset::default()
            .name("Moving Avg (20)")
            .marker(symbols::Marker::Braille)
            .style(Style::default().fg(Color::Yellow))
            .graph_type(GraphType::Line)
            .data(&moving_avg),
    ];

    let max_duration = state
        .episode_durations
        .iter()
        .map(|(_, d)| *d)
        .fold(100.0, f64::max);

    let chart = Chart::new(datasets)
        .block(
            Block::default()
                .title(" Episode Duration Over Time ")
                .title_style(
                    Style::default()
                        .fg(Color::Green)
                        .add_modifier(Modifier::BOLD),
                )
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Green)),
        )
        .x_axis(
            Axis::default()
                .title("Episode")
                .style(Style::default().fg(Color::Gray))
                .bounds([0.0, TOTAL_EPISODES as f64])
                .labels(vec![
                    Span::raw("0"),
                    Span::raw(format!("{}", TOTAL_EPISODES / 2)),
                    Span::raw(format!("{}", TOTAL_EPISODES)),
                ]),
        )
        .y_axis(
            Axis::default()
                .title("Steps")
                .style(Style::default().fg(Color::Gray))
                .bounds([0.0, max_duration * 1.1])
                .labels(vec![
                    Span::raw("0"),
                    Span::raw(format!("{:.0}", max_duration / 2.0)),
                    Span::raw(format!("{:.0}", max_duration)),
                ]),
        );

    frame.render_widget(chart, area);
}

fn render_epsilon_chart(frame: &mut Frame, area: Rect, state: &DashboardState) {
    // Generate theoretical epsilon decay curve
    let theoretical_curve: Vec<(f64, f64)> = (0..=10000)
        .step_by(100)
        .map(|steps| {
            let eps = EPS_END + (EPS_START - EPS_END) * (-1.0 * steps as f32 / EPS_DECAY).exp();
            (steps as f64, eps as f64)
        })
        .collect();

    // Actual epsilon values based on steps
    let actual_epsilon: Vec<(f64, f64)> = state
        .epsilon_history
        .iter()
        .enumerate()
        .map(|(i, (_, eps))| {
            let approx_steps = if i < state.episode_durations.len() {
                state.episode_durations[..=i]
                    .iter()
                    .map(|(_, d)| *d)
                    .sum::<f64>()
            } else {
                0.0
            };
            (approx_steps, *eps)
        })
        .collect();

    let datasets = vec![
        Dataset::default()
            .name("Theoretical e")
            .marker(symbols::Marker::Braille)
            .style(Style::default().fg(Color::DarkGray))
            .graph_type(GraphType::Line)
            .data(&theoretical_curve),
        Dataset::default()
            .name("Actual e")
            .marker(symbols::Marker::Dot)
            .style(Style::default().fg(Color::Magenta))
            .graph_type(GraphType::Scatter)
            .data(&actual_epsilon),
    ];

    let chart = Chart::new(datasets)
        .block(
            Block::default()
                .title(" Epsilon (Exploration Rate) Decay ")
                .title_style(
                    Style::default()
                        .fg(Color::Magenta)
                        .add_modifier(Modifier::BOLD),
                )
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Magenta)),
        )
        .x_axis(
            Axis::default()
                .title("Total Steps")
                .style(Style::default().fg(Color::Gray))
                .bounds([0.0, 10000.0])
                .labels(vec![
                    Span::raw("0"),
                    Span::raw("5000"),
                    Span::raw("10000"),
                ]),
        )
        .y_axis(
            Axis::default()
                .title("e")
                .style(Style::default().fg(Color::Gray))
                .bounds([0.0, 1.0])
                .labels(vec![Span::raw("0.0"), Span::raw("0.5"), Span::raw("1.0")]),
        );

    frame.render_widget(chart, area);
}

fn render_stats_panel(frame: &mut Frame, area: Rect, state: &DashboardState) {
    let stats_layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(8), // Current stats
            Constraint::Length(5), // Epsilon gauge
            Constraint::Length(5), // Progress gauge
            Constraint::Min(3),    // Hyperparameters
            Constraint::Min(3),    // Timings
        ])
        .split(area);

    // Current Statistics
    let stats_text = vec![
        Line::from(vec![
            Span::styled("Episode: ", Style::default().fg(Color::Gray)),
            Span::styled(
                format!("{}/{}", state.current_episode, TOTAL_EPISODES),
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            ),
        ]),
        Line::from(vec![
            Span::styled("Total Steps: ", Style::default().fg(Color::Gray)),
            Span::styled(format!("{}", state.total_steps), Style::default().fg(Color::White)),
        ]),
        Line::from(vec![
            Span::styled("Memory Size: ", Style::default().fg(Color::Gray)),
            Span::styled(
                format!("{}/{}", state.memory_size, MEMORY_CAPACITY),
                Style::default().fg(Color::Yellow),
            ),
        ]),
        Line::from(vec![
            Span::styled("Best Duration: ", Style::default().fg(Color::Gray)),
            Span::styled(
                format!("{} steps", state.best_duration),
                Style::default()
                    .fg(Color::Green)
                    .add_modifier(Modifier::BOLD),
            ),
        ]),
        Line::from(vec![
            Span::styled("Avg (100 ep): ", Style::default().fg(Color::Gray)),
            Span::styled(
                format!("{:.1} steps", state.avg_duration_100),
                Style::default().fg(Color::Blue),
            ),
        ]),
    ];

    let stats = Paragraph::new(stats_text).block(
        Block::default()
            .title(" Statistics ")
            .title_style(
                Style::default()
                    .fg(Color::White)
                    .add_modifier(Modifier::BOLD),
            )
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::White)),
    );
    frame.render_widget(stats, stats_layout[0]);

    // Epsilon Gauge
    let epsilon_percent =
        ((state.current_epsilon - EPS_END) / (EPS_START - EPS_END) * 100.0) as u16;
    let epsilon_gauge = Gauge::default()
        .block(
            Block::default()
                .title(format!(" Random Action: {:.1}% ", state.current_epsilon * 100.0))
                .title_style(Style::default().fg(Color::Magenta))
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Magenta)),
        )
        .gauge_style(
            Style::default()
                .fg(Color::Magenta)
                .bg(Color::DarkGray),
        )
        .percent(epsilon_percent.min(100))
        .label(format!("e = {:.3}", state.current_epsilon));
    frame.render_widget(epsilon_gauge, stats_layout[1]);

    // Progress Gauge
    let progress_percent =
        ((state.current_episode as f64 / TOTAL_EPISODES as f64) * 100.0) as u16;
    let progress_gauge = Gauge::default()
        .block(
            Block::default()
                .title(" Training Progress ")
                .title_style(Style::default().fg(Color::Green))
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Green)),
        )
        .gauge_style(Style::default().fg(Color::Green).bg(Color::DarkGray))
        .percent(progress_percent.min(100))
        .label(format!("{}%", progress_percent));
    frame.render_widget(progress_gauge, stats_layout[2]);

    // Hyperparameters
    let hyperparams = vec![
        Line::from(Span::styled(
            "--- Hyperparameters ---",
            Style::default().fg(Color::DarkGray),
        )),
        Line::from(format!("g={:.2}  t={:.3}  LR={:.0e}", GAMMA, TAU, LR)),
        Line::from(format!(
            "Batch={}  e:[{:.2},{:.2}]",
            BATCH_SIZE, EPS_END, EPS_START
        )),
        Line::from(format!("e_decay={:.0}", EPS_DECAY)),
    ];

    let mut total = 0;
    for (_, v) in &state.timing {
        total += v;
    }

    let mut timing_lines = vec![];
    for (k, v) in &state.timing {
        timing_lines.push(Line::from(format!(
            "{:?} {:?}",
            k,
            *v as f32 / total as f32
        )));
    }

    let hyperparams_widget = Paragraph::new(hyperparams)
        .style(Style::default().fg(Color::DarkGray))
        .alignment(Alignment::Center)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::DarkGray)),
        );

    let timings_block = Paragraph::new(timing_lines)
        .style(Style::default().fg(Color::DarkGray))
        .alignment(Alignment::Center)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::DarkGray)),
        );

    frame.render_widget(hyperparams_widget, stats_layout[3]);
    frame.render_widget(timings_block, stats_layout[4]);
}
