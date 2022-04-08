/*
 * Copyright (c) 2020 by Andrew Charneski.
 *
 * The author licenses this file to you under the
 * Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance
 * with the License.  You may obtain a copy
 * of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package com.simiacryptus.mindseye.test;

import com.simiacryptus.util.test.MacroTestRunner;
import com.simiacryptus.util.test.NotebookTestBase;
import org.junit.jupiter.api.Test;

import javax.annotation.Nonnull;

/**
 * This class contains all the local tests for the Java language.
 *
 * @docgenVersion 9
 */
public class LocalTests_Java extends NotebookTestBase {

  @Override
  public @Nonnull ReportType getReportType() {
    return ReportType.Components;
  }

  @Override
  protected Class<?> getTargetClass() {
    return MacroTestRunner.class;
  }

  /**
   * @Test public void main() {
   * new MacroTestRunner().runAll(getLog(), "com.simiacryptus.mindseye.layers");
   * }
   * @docgenVersion 9
   */
  @Test
  public void main() {
    new MacroTestRunner().runAll(getLog(),
        "com.simiacryptus.mindseye.layers"
    );
  }

}
