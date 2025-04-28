module     p2_gg_httbar_d260h8l132_qp
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity8d260h8l132_qp.f90
   ! generator: buildfortran_tn3.py
   use p2_gg_httbar_config, only: ki => ki_qp
   use p2_gg_httbar_util_qp, only: cond_t, d => metric_tensor
   implicit none
   private
   complex(ki), parameter :: i_ = (0.0_ki, 1.0_ki)
   integer, parameter :: ninjaidxt1x0mu0 = 0
   integer, parameter :: ninjaidxt0x0mu0 = 1
   integer, parameter :: ninjaidxt0x1mu0 = 2
   public :: numerator_t2
contains
!---#[ subroutine brack_21:
   pure subroutine brack_21(ninjaA0, ninjaA1, ninjaE3, ninjaE4, ninjaP, brack)
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_color_qp
      use p2_gg_httbar_abbrevd260h8_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA0, ninjaA1, ninjaE3, ninja&
      &E4
      complex(ki), dimension(0:2), intent(in) :: ninjaP
      complex(ki), dimension(1) :: acd260
      complex(ki), dimension (0:*), intent(inout) :: brack
      complex(ki) :: ninjaP0, ninjaP1, ninjaP2
      ninjaP0 = ninjaP(0)
      ninjaP1 = ninjaP(1)
      ninjaP2 = ninjaP(2)
      brack(ninjaidxt1x0mu0)=0.0_ki
   end subroutine brack_21
!---#] subroutine brack_21:
!---#[ subroutine brack_22:
   pure subroutine brack_22(ninjaA0, ninjaA1, ninjaE3, ninjaE4, ninjaP, brack)
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_color_qp
      use p2_gg_httbar_abbrevd260h8_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA0, ninjaA1, ninjaE3, ninja&
      &E4
      complex(ki), dimension(0:2), intent(in) :: ninjaP
      complex(ki), dimension(34) :: acd260
      complex(ki), dimension (0:*), intent(inout) :: brack
      complex(ki) :: ninjaP0, ninjaP1, ninjaP2
      ninjaP0 = ninjaP(0)
      ninjaP1 = ninjaP(1)
      ninjaP2 = ninjaP(2)
      acd260(1)=dotproduct(k2,ninjaE3)
      acd260(2)=dotproduct(ninjaE3,spvak2e1)
      acd260(3)=dotproduct(ninjaE3,spvae1e2)
      acd260(4)=abb260(34)
      acd260(5)=dotproduct(e2,ninjaE3)
      acd260(6)=dotproduct(ninjaA0,ninjaE3)
      acd260(7)=abb260(45)
      acd260(8)=dotproduct(ninjaE3,spvae1k2)
      acd260(9)=abb260(44)
      acd260(10)=dotproduct(ninjaE3,spval4e1)
      acd260(11)=abb260(51)
      acd260(12)=dotproduct(ninjaE3,spvae1l5)
      acd260(13)=abb260(87)
      acd260(14)=abb260(10)
      acd260(15)=dotproduct(ninjaE3,spvae1l3)
      acd260(16)=abb260(56)
      acd260(17)=dotproduct(ninjaE3,spval3e1)
      acd260(18)=abb260(39)
      acd260(19)=abb260(142)
      acd260(20)=abb260(134)
      acd260(21)=abb260(112)
      acd260(22)=abb260(85)
      acd260(23)=dotproduct(ninjaE3,spvae2e1)
      acd260(24)=abb260(73)
      acd260(25)=dotproduct(ninjaE3,spvak2l5)
      acd260(26)=abb260(27)
      acd260(27)=dotproduct(ninjaE3,spval3l5)
      acd260(28)=dotproduct(ninjaE3,spvak2l3)
      acd260(29)=acd260(10)*acd260(11)
      acd260(30)=acd260(12)*acd260(13)
      acd260(31)=acd260(8)*acd260(9)
      acd260(32)=acd260(2)*acd260(7)
      acd260(29)=acd260(32)+acd260(31)+acd260(29)+acd260(30)
      acd260(30)=2.0_ki*acd260(6)
      acd260(29)=acd260(29)*acd260(30)
      acd260(31)=acd260(17)*acd260(19)
      acd260(32)=acd260(10)*acd260(21)
      acd260(31)=acd260(31)+acd260(32)
      acd260(31)=acd260(12)*acd260(31)
      acd260(32)=acd260(15)*acd260(16)
      acd260(33)=acd260(8)*acd260(14)
      acd260(32)=acd260(32)+acd260(33)
      acd260(32)=acd260(2)*acd260(32)
      acd260(33)=acd260(10)*acd260(15)*acd260(20)
      acd260(34)=acd260(8)*acd260(17)*acd260(18)
      acd260(29)=acd260(29)+acd260(32)+acd260(34)+acd260(33)+acd260(31)
      acd260(29)=acd260(5)*acd260(29)
      acd260(31)=acd260(23)*acd260(24)
      acd260(32)=acd260(12)*acd260(31)
      acd260(33)=acd260(2)*acd260(3)
      acd260(34)=acd260(22)*acd260(33)
      acd260(32)=acd260(32)+acd260(34)
      acd260(30)=acd260(32)*acd260(30)
      acd260(31)=-acd260(15)*acd260(27)*acd260(31)
      acd260(32)=-acd260(17)*acd260(28)*acd260(3)*acd260(22)
      acd260(34)=acd260(8)*acd260(23)*acd260(25)*acd260(26)
      acd260(33)=acd260(1)*acd260(4)*acd260(33)
      acd260(29)=acd260(29)+acd260(30)+acd260(33)+acd260(34)+acd260(31)+acd260(&
      &32)
      brack(ninjaidxt0x0mu0)=acd260(29)
      brack(ninjaidxt0x1mu0)=0.0_ki
   end subroutine brack_22
!---#] subroutine brack_22:
!---#[ subroutine numerator_t2:
   subroutine numerator_t2(ncut, a0, a1, b, c, param, deg, coeffs) &
   & bind(c, name="p2_gg_httbar_d260h8_qp_ninja_t2")
      use iso_c_binding, only: c_int
      use quadninjago_module, only: ki => ki_nin
      use p2_gg_httbar_globalsl1_qp, only: epspow
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_abbrevd260h8_qp
      implicit none
      integer(c_int), intent(in) :: ncut, deg
      complex(ki), dimension(0:3), intent(in) :: a0, a1, b, c
      complex(ki), dimension(0:2), intent(in) :: param
      complex(ki), dimension(4) :: vecA0, vecA1, vecB, vecC
      complex(ki), dimension(0:*), intent(out) :: coeffs
      integer :: t1
      complex(ki), dimension(4) :: qshift
      qshift = k2-k3-k5
      vecA0(1:4) = + a0(0:3) - qshift(1:4)
      vecA1(1:4) = + a1(0:3)
      vecB(1:4) = + b(0:3)
      vecC(1:4) = + c(0:3)
      if (deg.lt.0) return
      t1 = 0
      call cond_t(epspow.eq.t1,brack_21,vecA0,vecA1,vecB,vecC,param,coeffs)
      if (deg.le.(1+(-1))) return
      call cond_t(epspow.eq.t1,brack_22,vecA0,vecA1,vecB,vecC,param,coeffs)
   end subroutine numerator_t2
!---#] subroutine numerator_t2:
end module     p2_gg_httbar_d260h8l132_qp
