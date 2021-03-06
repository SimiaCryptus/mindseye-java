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

import com.google.gson.JsonObject;
import com.simiacryptus.mindseye.lang.Layer;
import com.simiacryptus.mindseye.network.DAGNetwork;
import com.simiacryptus.mindseye.network.DAGNode;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Map;
import java.util.UUID;

/**
 * The type Sign reducer layer.
 */
@SuppressWarnings("serial")
public class SignReducerLayer extends DAGNetwork {

  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(SignReducerLayer.class);
  @Nullable
  private final DAGNode head;

  /**
   * Instantiates a new Sign reducer layer.
   */
  public SignReducerLayer() {
    super(1);
    final DAGNode avgInput = add(new AvgReducerLayer(), getInput(0));
    SigmoidActivationLayer sigmoid = new SigmoidActivationLayer();
    NthPowerActivationLayer inv = new NthPowerActivationLayer();
    NthPowerActivationLayer sqrt = new NthPowerActivationLayer();
    LinearActivationLayer negative = new LinearActivationLayer();
    negative.setScale(-1);
    sqrt.setPower(0.5);
    inv.setPower(-1);
    sigmoid.setBalanced(false);
    head = add(sigmoid,
        add(new ProductInputsLayer(),
            avgInput.addRef(),
            add(inv,
                add(sqrt,
                    add(new SumInputsLayer(),
                        add(new AvgReducerLayer(),
                            add(new SqActivationLayer(), getInput(0))),
                        add(negative,
                            add(new SqActivationLayer(), avgInput.addRef())))))));
    avgInput.freeRef();
  }

  /**
   * Instantiates a new Sign reducer layer.
   *
   * @param json the json
   * @param rs   the rs
   */
  protected SignReducerLayer(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    super(json, rs);
    head = getNodeById(UUID.fromString(json.getAsJsonPrimitive("head").getAsString()));
  }

  @Override
  public DAGNode getHead() {
    return head == null ? null : head.addRef();
  }

  /**
   * From json layer.
   *
   * @param inner the inner
   * @param rs    the rs
   * @return the layer
   */
  @Nonnull
  @SuppressWarnings("unused")
  public static Layer fromJson(@Nonnull final JsonObject inner, Map<CharSequence, byte[]> rs) {
    return new SignReducerLayer(inner, rs);
  }

  public void _free() {
    if (null != head)
      head.freeRef();
    super._free();
  }

  @Nonnull
  public @Override
  @SuppressWarnings("unused")
  SignReducerLayer addRef() {
    return (SignReducerLayer) super.addRef();
  }
}
