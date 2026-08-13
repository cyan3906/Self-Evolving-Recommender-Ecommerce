from __future__ import annotations

import hashlib
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

from writer import hybrid_retrieval
from store import  save_hybird_retrieval

# ============================================================
# 配置
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

DATABASE_DIR = PROJECT_ROOT / "db"
DATABASE_PATH = DATABASE_DIR / "sqlite/skills.db"


# ============================================================
# Skill 数据库
# ============================================================

class SkillDatabase:

    def __init__(
        self,
        database_path: str | Path = DATABASE_PATH,
    ):
        self.database_path = Path(database_path)

        self.database_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.create_table()

    # ========================================================
    # 获取数据库连接
    # ========================================================

    def _get_connection(self) -> sqlite3.Connection:

        conn = sqlite3.connect(
            str(self.database_path)
        )

        conn.row_factory = sqlite3.Row

        return conn

    # ========================================================
    # 创建表
    # ========================================================

    def create_table(self):

        sql = """
        CREATE TABLE IF NOT EXISTS skill_registry (
            id INTEGER PRIMARY KEY AUTOINCREMENT,

            file_name TEXT NOT NULL UNIQUE,

            skill_name TEXT NOT NULL,

            skill_type TEXT NOT NULL,

            description TEXT,

            file_path TEXT,

            use_count INTEGER NOT NULL DEFAULT 0,

            success_count INTEGER NOT NULL DEFAULT 0,

            failure_count INTEGER NOT NULL DEFAULT 0,

            last_used_at TEXT,

            status TEXT NOT NULL DEFAULT 'active',

            version INTEGER NOT NULL DEFAULT 1,

            content_hash TEXT,

            created_at TEXT NOT NULL,

            updated_at TEXT NOT NULL
        );
        """

        with self._get_connection() as conn:

            conn.execute(sql)

            # skill 类型索引
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_skill_type
                ON skill_registry(skill_type);
                """
            )

            # 最近使用时间索引
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_last_used_at
                ON skill_registry(last_used_at);
                """
            )

            # 状态索引
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_skill_status
                ON skill_registry(status);
                """
            )

            conn.commit()

    # ========================================================
    # 创建 Skill
    # ========================================================

    def create_skill(
        self,
        file_name: str,
        skill_name: str,
        skill_type: str,
        description: Optional[str] = None,
        file_path: Optional[str] = None,
        content: Optional[str] = None,
    ) -> int:

        now = self._now()

        content_hash = None

        if content:
            content_hash = self._calculate_hash(content)

        sql = """
        INSERT INTO skill_registry (
            file_name,
            skill_name,
            skill_type,
            description,
            file_path,
            use_count,
            success_count,
            failure_count,
            last_used_at,
            status,
            version,
            content_hash,
            created_at,
            updated_at
        )
        VALUES (?, ?, ?, ?, ?, 0, 0, 0, NULL, 'active', 1, ?, ?, ?)
        """

        with self._get_connection() as conn:

            cursor = conn.execute(
                sql,
                (
                    file_name,
                    skill_name,
                    skill_type,
                    description,
                    file_path,
                    content_hash,
                    now,
                    now,
                ),
            )

            conn.commit()

            return cursor.lastrowid

    # ========================================================
    # Skill 是否存在
    # ========================================================

    def exists(
        self,
        file_name: str,
    ) -> bool:

        sql = """
        SELECT 1
        FROM skill_registry
        WHERE file_name = ?
        LIMIT 1
        """

        with self._get_connection() as conn:

            row = conn.execute(
                sql,
                (file_name,),
            ).fetchone()

        return row is not None

    # ========================================================
    # 查询单个 Skill
    # ========================================================

    def get_skill(
        self,
        file_name: str,
    ) -> Optional[dict]:

        sql = """
        SELECT *
        FROM skill_registry
        WHERE file_name = ?
        """

        with self._get_connection() as conn:

            row = conn.execute(
                sql,
                (file_name,),
            ).fetchone()

        if row is None:
            return None

        return dict(row)

    # ========================================================
    # 查询所有 Skill
    # ========================================================

    def get_all_skills(
        self,
        only_active: bool = True,
    ) -> list[dict]:

        if only_active:

            sql = """
            SELECT *
            FROM skill_registry
            WHERE status = 'active'
            ORDER BY updated_at DESC
            """

        else:

            sql = """
            SELECT *
            FROM skill_registry
            ORDER BY updated_at DESC
            """

        with self._get_connection() as conn:

            rows = conn.execute(sql).fetchall()

        return [
            dict(row)
            for row in rows
        ]

    # ========================================================
    # 按 Skill 类型查询
    # ========================================================

    def get_skills_by_type(
        self,
        skill_type: str,
    ) -> list[dict]:

        sql = """
        SELECT *
        FROM skill_registry
        WHERE skill_type = ?
          AND status = 'active'
        ORDER BY use_count DESC
        """

        with self._get_connection() as conn:

            rows = conn.execute(
                sql,
                (skill_type,),
            ).fetchall()

        return [
            dict(row)
            for row in rows
        ]

    # ========================================================
    # 记录 Skill 使用
    # ========================================================

    def record_usage(
        self,
        file_name: str,
        success: bool = True,
    ) -> bool:
        """
        Skill 每执行一次调用这个方法。

        success=True:
            use_count + 1
            success_count + 1

        success=False:
            use_count + 1
            failure_count + 1

        同时更新 last_used_at。
        """

        now = self._now()

        if success:

            sql = """
            UPDATE skill_registry
            SET
                use_count = use_count + 1,
                success_count = success_count + 1,
                last_used_at = ?,
                updated_at = ?
            WHERE file_name = ?
            """

        else:

            sql = """
            UPDATE skill_registry
            SET
                use_count = use_count + 1,
                failure_count = failure_count + 1,
                last_used_at = ?,
                updated_at = ?
            WHERE file_name = ?
            """

        with self._get_connection() as conn:

            cursor = conn.execute(
                sql,
                (
                    now,
                    now,
                    file_name,
                ),
            )

            conn.commit()

        return cursor.rowcount > 0

    # ========================================================
    # 更新 Skill
    # ========================================================

    def update_skill(
        self,
        file_name: str,
        skill_type: Optional[str] = None,
        description: Optional[str] = None,
        file_path: Optional[str] = None,
        content: Optional[str] = None,
    ) -> bool:

        skill = self.get_skill(file_name)

        if skill is None:
            return False

        now = self._now()

        new_skill_type = (
            skill_type
            if skill_type is not None
            else skill["skill_type"]
        )

        new_description = (
            description
            if description is not None
            else skill["description"]
        )

        new_file_path = (
            file_path
            if file_path is not None
            else skill["file_path"]
        )

        content_hash = skill["content_hash"]

        version = skill["version"]

        # 如果 Skill 内容发生改变，版本 +1
        if content is not None:

            new_hash = self._calculate_hash(content)

            if new_hash != content_hash:
                content_hash = new_hash
                version += 1

        sql = """
        UPDATE skill_registry
        SET
            skill_type = ?,
            description = ?,
            file_path = ?,
            content_hash = ?,
            version = ?,
            updated_at = ?
        WHERE file_name = ?
        """

        with self._get_connection() as conn:

            cursor = conn.execute(
                sql,
                (
                    new_skill_type,
                    new_description,
                    new_file_path,
                    content_hash,
                    version,
                    now,
                    file_name,
                ),
            )

            conn.commit()

        return cursor.rowcount > 0

    # ========================================================
    # 修改状态
    # ========================================================

    def update_status(
        self,
        file_name: str,
        status: str,
    ) -> bool:

        allowed_status = {
            "active",
            "disabled",
            "deprecated",
        }

        if status not in allowed_status:
            raise ValueError(
                f"不支持的 status: {status}"
            )

        sql = """
        UPDATE skill_registry
        SET
            status = ?,
            updated_at = ?
        WHERE file_name = ?
        """

        with self._get_connection() as conn:

            cursor = conn.execute(
                sql,
                (
                    status,
                    self._now(),
                    file_name,
                ),
            )

            conn.commit()

        return cursor.rowcount > 0

    # ========================================================
    # 获取最常使用 Skill
    # ========================================================

    def get_most_used_skills(
        self,
        limit: int = 10,
    ) -> list[dict]:

        sql = """
        SELECT *
        FROM skill_registry
        WHERE status = 'active'
        ORDER BY use_count DESC
        LIMIT ?
        """

        with self._get_connection() as conn:

            rows = conn.execute(
                sql,
                (limit,),
            ).fetchall()

        return [
            dict(row)
            for row in rows
        ]

    # ========================================================
    # 工具方法
    # ========================================================

    @staticmethod
    def _now() -> str:

        return datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )

    @staticmethod
    def _calculate_hash(
        content: str,
    ) -> str:

        return hashlib.sha256(
            content.encode("utf-8")
        ).hexdigest()


# ============================================================
# 测试
# ============================================================

if __name__ == "__main__":

    skill_db = SkillDatabase()

    # --------------------------------------------------------
    # 创建 Skill
    # --------------------------------------------------------

    file_name = "hybrid-retrieval.md"

    res = save_hybird_retrieval(
        file_name="SKILL2",
        potential_products=["商品1", "商品2", "商品3"],
        es_top_k=10,
        milvus_top_k=10,
        final_top_k=10,
        rrf_k=10,
        es_weight=0.7,
        milvus_weight=0.3,
    )
    
    print(111,res)
    
    if not skill_db.exists(file_name):

        skill_id = skill_db.create_skill(
            file_name=file_name,

            skill_name="hybrid-retrieval",

            skill_type="混合检索",

            description=(
                "规定 Elasticsearch 与 Milvus "
                "按照 0.7 / 0.3 比例进行混合检索"
            ),

            file_path=(
                "./skills/hybrid-retrieval/"
                "SKILL.md"
            ),

            content=content,
        )

        print(
            f"Skill 创建成功，ID={skill_id}"
        )

    print("点我继续")
    # input()
    
    # --------------------------------------------------------
    # 模拟调用成功
    # --------------------------------------------------------

    skill_db.record_usage(
        file_name="hybrid-retrieval.md",
        success=True,
    )

    # --------------------------------------------------------
    # 查询 Skill
    # --------------------------------------------------------

    skill = skill_db.get_skill(
        "hybrid-retrieval.md"
    )

    print(skill)

    # --------------------------------------------------------
    # 查询所有 Skill
    # --------------------------------------------------------

    skills = skill_db.get_all_skills()

    for item in skills:
        print(item)