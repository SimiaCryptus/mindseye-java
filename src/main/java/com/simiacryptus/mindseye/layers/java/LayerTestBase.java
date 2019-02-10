/*
 * Copyright (c) 2019 by Andrew Charneski.
 *
 * The author licenses this file to you under the
 * Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance
 * with the License.  You may obtain a copy
 * of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package com.simiacryptus.mindseye.layers.java;

import com.simiacryptus.mindseye.lang.Layer;
import com.simiacryptus.mindseye.test.unit.StandardLayerTests;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

/**
 * The type LayerBase apply base.
 */
public abstract class LayerTestBase extends StandardLayerTests {

  @Override
  protected Layer lossLayer() {
    return new EntropyLossLayer();
  }

  /**
   * Test.
   *
   */
  @Test(timeout = 15 * 60 * 1000)
  public void test() {
    run(this::run);
  }

//  @Test(timeout = 15 * 60 * 1000)
//  public void testMonteCarlo() throws Throwable {
//    apply(this::monteCarlo);
//  }

  /**
   * Clean all.
   */
  @Before
  public void setup() {
    reportingFolder = "reports/_reports";
    //GpuController.remove();
  }

  /**
   * Cleanup.
   */
  @After
  public void cleanup() {
    System.gc();
    //GpuController.remove();
  }

}
