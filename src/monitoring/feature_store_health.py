import sqlite3
import os
import yaml
import time
import logging

logger = logging.getLogger('sentinel.monitoring')


def check_feature_store_health(quiet: bool = False) -> bool:
    """Check whether the offline feature store is fresh enough.

    Args:
        quiet: If True, suppress log output (used by health endpoint).

    Returns:
        True if healthy, False otherwise.
    """
    base_dir = os.path.dirname(__file__)
    data_dir = os.path.join(base_dir, '..', '..', 'data')
    config_dir = os.path.join(base_dir, '..', '..', 'config')

    with open(os.path.join(config_dir, 'cost_matrix.yaml'), 'r') as f:
        config = yaml.safe_load(f)

    refresh_hours = config['batch_job']['ring_score_refresh_hours']
    multiplier = config['batch_job']['staleness_alert_multiplier']
    max_age_seconds = refresh_hours * 3600 * multiplier

    db_path = os.path.join(data_dir, 'feature_store.db')

    if not os.path.exists(db_path):
        if not quiet:
            logger.warning('Feature store DB does not exist at %s', db_path)
        return False

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT last_run FROM batch_metadata')
        row = cursor.fetchone()
        conn.close()

        if not row:
            if not quiet:
                logger.warning('No batch metadata found in feature store.')
            return False

        last_run = row[0]
        age = time.time() - last_run

        if age > max_age_seconds:
            if not quiet:
                logger.warning(
                    'Feature store stale: last run %.2f hours ago (max %.2f)',
                    age / 3600,
                    max_age_seconds / 3600,
                )
            return False

        if not quiet:
            logger.info('Feature store healthy: last refresh %.2f hours ago.', age / 3600)
        return True

    except Exception as e:
        if not quiet:
            logger.exception('Could not check feature store health: %s', e)
        return False


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    check_feature_store_health()
