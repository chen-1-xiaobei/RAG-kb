-- 报销提案与审计的 MySQL 表结构。
--
-- 应用启动时会自动建表（mysql_claims.py 里的 SCHEMA_STATEMENTS），
-- 这份文件是给 DBA/部署核对用的：两边必须一致，改一处就要改另一处。
--
-- 与 SQLite 版的差异（刻意为之）：
--   1. AUTOINCREMENT -> AUTO_INCREMENT，且用 BIGINT
--   2. 时间戳仍是 VARCHAR(32) 存 ISO-8601 字符串，不用 DATETIME
--      理由见 mysql_claims.py 顶部：避免时区语义，且 Proposal.created_at 保持 str
--   3. TEXT 列不能有 DEFAULT，detail 的默认值由应用侧传空串
--   4. 显式声明 InnoDB + utf8mb4（SQLite 没有这些概念）
--
-- 并发语义与应用层一致：提交靠单条
--   UPDATE ... SET status='committed' WHERE ... AND status='pending'
-- 的 rowcount 判定，不依赖先读后写。

CREATE TABLE IF NOT EXISTS proposals (
    proposal_id     VARCHAR(64)  NOT NULL,
    principal_id    VARCHAR(128) NOT NULL,
    conversation_id VARCHAR(64)  NOT NULL,
    status          VARCHAR(16)  NOT NULL COMMENT 'pending | committed',
    claim_json      TEXT         NOT NULL COMMENT 'Claim.as_dict() 的 JSON',
    needs_dual_sign TINYINT(1)   NOT NULL DEFAULT 0,
    created_at      VARCHAR(32)  NOT NULL COMMENT 'ISO-8601',
    committed_at    VARCHAR(32)  NULL COMMENT 'ISO-8601',
    PRIMARY KEY (proposal_id),
    KEY idx_proposal_owner (principal_id, conversation_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

CREATE TABLE IF NOT EXISTS claim_audit (
    id              BIGINT AUTO_INCREMENT PRIMARY KEY,
    at              VARCHAR(32)  NOT NULL COMMENT 'ISO-8601',
    principal_id    VARCHAR(128) NOT NULL,
    conversation_id VARCHAR(64)  NOT NULL,
    proposal_id     VARCHAR(64)  NOT NULL COMMENT '被拒的提案为空串',
    action          VARCHAR(16)  NOT NULL COMMENT 'propose | commit',
    result          VARCHAR(32)  NOT NULL,
    detail          TEXT         NOT NULL,
    KEY idx_audit_proposal (proposal_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
