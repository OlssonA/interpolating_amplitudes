module     p2_gg_httbar_d263h0l132
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity0d263h0l132.f90
   ! generator: buildfortran_tn3.py
   use p2_gg_httbar_config, only: ki
   use p2_gg_httbar_util, only: cond_t, d => metric_tensor
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
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd263h0
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA0, ninjaA1, ninjaE3, ninja&
      &E4
      complex(ki), dimension(0:2), intent(in) :: ninjaP
      complex(ki), dimension(1) :: acd263
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
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd263h0
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA0, ninjaA1, ninjaE3, ninja&
      &E4
      complex(ki), dimension(0:2), intent(in) :: ninjaP
      complex(ki), dimension(33) :: acd263
      complex(ki), dimension (0:*), intent(inout) :: brack
      complex(ki) :: ninjaP0, ninjaP1, ninjaP2
      ninjaP0 = ninjaP(0)
      ninjaP1 = ninjaP(1)
      ninjaP2 = ninjaP(2)
      acd263(1)=dotproduct(k2,ninjaE3)
      acd263(2)=dotproduct(ninjaE3,spvae2k2)
      acd263(3)=dotproduct(ninjaE3,spvae1e2)
      acd263(4)=abb263(9)
      acd263(5)=dotproduct(l5,ninjaE3)
      acd263(6)=dotproduct(ninjaE3,spval5e2)
      acd263(7)=dotproduct(ninjaE3,spvae2e1)
      acd263(8)=abb263(11)
      acd263(9)=dotproduct(e1,ninjaE3)
      acd263(10)=dotproduct(ninjaA0,ninjaE3)
      acd263(11)=abb263(150)
      acd263(12)=dotproduct(ninjaE3,spval4e2)
      acd263(13)=abb263(74)
      acd263(14)=abb263(38)
      acd263(15)=abb263(28)
      acd263(16)=dotproduct(ninjaE3,spvae2k1)
      acd263(17)=abb263(13)
      acd263(18)=dotproduct(ninjaE3,spvae2l5)
      acd263(19)=abb263(201)
      acd263(20)=abb263(93)
      acd263(21)=abb263(44)
      acd263(22)=abb263(199)
      acd263(23)=dotproduct(ninjaE3,spvak1k2)
      acd263(24)=dotproduct(ninjaE3,spval5k2)
      acd263(25)=dotproduct(ninjaE3,spvak2e2)
      acd263(26)=abb263(83)
      acd263(27)=dotproduct(ninjaE3,spval5k1)
      acd263(28)=dotproduct(ninjaE3,spvak1e2)
      acd263(29)=acd263(18)*acd263(19)
      acd263(30)=acd263(16)*acd263(17)
      acd263(31)=2.0_ki*acd263(10)
      acd263(32)=acd263(11)*acd263(31)
      acd263(33)=acd263(2)*acd263(14)
      acd263(29)=acd263(33)+acd263(32)+acd263(29)+acd263(30)
      acd263(29)=acd263(6)*acd263(29)
      acd263(30)=acd263(16)*acd263(20)
      acd263(32)=acd263(13)*acd263(31)
      acd263(33)=acd263(2)*acd263(15)
      acd263(30)=acd263(33)+acd263(30)+acd263(32)
      acd263(30)=acd263(12)*acd263(30)
      acd263(29)=acd263(29)+acd263(30)
      acd263(29)=acd263(9)*acd263(29)
      acd263(30)=acd263(5)*acd263(8)
      acd263(32)=acd263(22)*acd263(31)
      acd263(30)=acd263(30)+acd263(32)
      acd263(30)=acd263(6)*acd263(30)
      acd263(32)=acd263(24)*acd263(25)*acd263(26)
      acd263(33)=-acd263(22)*acd263(27)*acd263(28)
      acd263(30)=acd263(30)+acd263(32)+acd263(33)
      acd263(30)=acd263(7)*acd263(30)
      acd263(32)=-acd263(18)*acd263(24)
      acd263(33)=acd263(16)*acd263(23)
      acd263(32)=acd263(32)+acd263(33)
      acd263(32)=acd263(21)*acd263(32)
      acd263(33)=-acd263(1)*acd263(4)
      acd263(31)=-acd263(21)*acd263(31)
      acd263(31)=acd263(33)+acd263(31)
      acd263(31)=acd263(2)*acd263(31)
      acd263(31)=acd263(31)+acd263(32)
      acd263(31)=acd263(3)*acd263(31)
      acd263(29)=acd263(29)+acd263(31)+acd263(30)
      brack(ninjaidxt0x0mu0)=acd263(29)
      brack(ninjaidxt0x1mu0)=0.0_ki
   end subroutine brack_22
!---#] subroutine brack_22:
!---#[ subroutine numerator_t2:
   subroutine numerator_t2(ncut, a0, a1, b, c, param, deg, coeffs) &
   & bind(c, name="p2_gg_httbar_d263h0_ninja_t2")
      use iso_c_binding, only: c_int
      use ninjago_module, only: ki => ki_nin
      use p2_gg_httbar_globalsl1, only: epspow
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_abbrevd263h0
      implicit none
      integer(c_int), intent(in) :: ncut, deg
      complex(ki), dimension(0:3), intent(in) :: a0, a1, b, c
      complex(ki), dimension(0:2), intent(in) :: param
      complex(ki), dimension(4) :: vecA0, vecA1, vecB, vecC
      complex(ki), dimension(0:*), intent(out) :: coeffs
      integer :: t1
      vecA0(1:4) = + a0(0:3)
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
end module     p2_gg_httbar_d263h0l132
