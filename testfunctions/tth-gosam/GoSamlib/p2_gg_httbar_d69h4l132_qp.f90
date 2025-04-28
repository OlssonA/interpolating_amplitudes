module     p2_gg_httbar_d69h4l132_qp
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity4d69h4l132_qp.f90
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
      use p2_gg_httbar_abbrevd69h4_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA0, ninjaA1, ninjaE3, ninja&
      &E4
      complex(ki), dimension(0:2), intent(in) :: ninjaP
      complex(ki), dimension(1) :: acd69
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
      use p2_gg_httbar_abbrevd69h4_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA0, ninjaA1, ninjaE3, ninja&
      &E4
      complex(ki), dimension(0:2), intent(in) :: ninjaP
      complex(ki), dimension(60) :: acd69
      complex(ki), dimension (0:*), intent(inout) :: brack
      complex(ki) :: ninjaP0, ninjaP1, ninjaP2
      ninjaP0 = ninjaP(0)
      ninjaP1 = ninjaP(1)
      ninjaP2 = ninjaP(2)
      acd69(1)=dotproduct(k2,ninjaE3)
      acd69(2)=dotproduct(ninjaE3,spvae2k2)
      acd69(3)=abb69(18)
      acd69(4)=dotproduct(ninjaA0,ninjaE3)
      acd69(5)=abb69(14)
      acd69(6)=dotproduct(ninjaE3,spval4e2)
      acd69(7)=abb69(28)
      acd69(8)=dotproduct(ninjaE3,spval5e2)
      acd69(9)=abb69(15)
      acd69(10)=dotproduct(ninjaE3,spvae2e1)
      acd69(11)=abb69(47)
      acd69(12)=dotproduct(ninjaE3,spvae1e2)
      acd69(13)=abb69(43)
      acd69(14)=dotproduct(ninjaE3,spvak1e2)
      acd69(15)=abb69(32)
      acd69(16)=dotproduct(ninjaE3,spvae2l4)
      acd69(17)=abb69(42)
      acd69(18)=dotproduct(ninjaE3,spvae2k1)
      acd69(19)=abb69(44)
      acd69(20)=abb69(9)
      acd69(21)=abb69(24)
      acd69(22)=abb69(21)
      acd69(23)=abb69(22)
      acd69(24)=dotproduct(ninjaE3,spvak2e1)
      acd69(25)=abb69(27)
      acd69(26)=dotproduct(ninjaE3,spvak2l4)
      acd69(27)=abb69(35)
      acd69(28)=dotproduct(ninjaE3,spvak2k1)
      acd69(29)=abb69(37)
      acd69(30)=dotproduct(ninjaE3,spval3k2)
      acd69(31)=dotproduct(ninjaE3,spvae2l3)
      acd69(32)=dotproduct(ninjaE3,spval3l4)
      acd69(33)=dotproduct(ninjaE3,spval3k1)
      acd69(34)=dotproduct(ninjaE3,spval3e1)
      acd69(35)=abb69(20)
      acd69(36)=abb69(48)
      acd69(37)=abb69(50)
      acd69(38)=dotproduct(ninjaE3,spval5l3)
      acd69(39)=dotproduct(ninjaE3,spval3e2)
      acd69(40)=dotproduct(ninjaE3,spval4l3)
      acd69(41)=dotproduct(ninjaE3,spvak1l3)
      acd69(42)=dotproduct(ninjaE3,spvae1l3)
      acd69(43)=dotproduct(ninjaE3,spvae1k2)
      acd69(44)=dotproduct(ninjaE3,spvak2e2)
      acd69(45)=abb69(25)
      acd69(46)=dotproduct(ninjaE3,spval5k2)
      acd69(47)=abb69(30)
      acd69(48)=dotproduct(ninjaE3,spval4k2)
      acd69(49)=abb69(33)
      acd69(50)=dotproduct(ninjaE3,spvak1k2)
      acd69(51)=abb69(34)
      acd69(52)=acd69(3)*acd69(1)
      acd69(53)=acd69(20)*acd69(6)
      acd69(54)=acd69(21)*acd69(8)
      acd69(55)=acd69(22)*acd69(12)
      acd69(56)=acd69(23)*acd69(14)
      acd69(57)=acd69(25)*acd69(24)
      acd69(58)=acd69(27)*acd69(26)
      acd69(59)=acd69(29)*acd69(28)
      acd69(52)=acd69(59)+acd69(58)+acd69(57)+acd69(56)+acd69(55)+acd69(54)+acd&
      &69(53)+acd69(52)
      acd69(52)=acd69(2)*acd69(52)
      acd69(53)=acd69(5)*acd69(2)
      acd69(54)=-acd69(7)*acd69(6)
      acd69(55)=acd69(9)*acd69(8)
      acd69(56)=acd69(11)*acd69(10)
      acd69(57)=-acd69(13)*acd69(12)
      acd69(58)=acd69(15)*acd69(14)
      acd69(59)=-acd69(17)*acd69(16)
      acd69(60)=acd69(19)*acd69(18)
      acd69(53)=acd69(60)+acd69(59)+acd69(58)+acd69(57)+acd69(56)+acd69(55)+acd&
      &69(53)+acd69(54)
      acd69(53)=acd69(4)*acd69(53)
      acd69(54)=-acd69(30)*acd69(5)
      acd69(55)=acd69(32)*acd69(17)
      acd69(56)=-acd69(33)*acd69(19)
      acd69(57)=-acd69(34)*acd69(11)
      acd69(54)=acd69(57)+acd69(56)+acd69(55)+acd69(54)
      acd69(54)=acd69(31)*acd69(54)
      acd69(55)=-acd69(38)*acd69(9)
      acd69(56)=acd69(40)*acd69(7)
      acd69(57)=-acd69(41)*acd69(15)
      acd69(58)=acd69(42)*acd69(13)
      acd69(55)=acd69(58)+acd69(57)+acd69(56)+acd69(55)
      acd69(55)=acd69(39)*acd69(55)
      acd69(56)=acd69(45)*acd69(43)
      acd69(57)=acd69(47)*acd69(46)
      acd69(58)=acd69(49)*acd69(48)
      acd69(59)=acd69(51)*acd69(50)
      acd69(56)=acd69(59)+acd69(58)+acd69(57)+acd69(56)
      acd69(56)=acd69(44)*acd69(56)
      acd69(57)=acd69(35)*acd69(10)
      acd69(58)=acd69(36)*acd69(16)
      acd69(59)=-acd69(37)*acd69(18)
      acd69(57)=acd69(59)+acd69(58)+acd69(57)
      acd69(57)=acd69(8)*acd69(57)
      acd69(52)=2.0_ki*acd69(53)+acd69(52)+acd69(56)+acd69(55)+acd69(54)+acd69(&
      &57)
      brack(ninjaidxt0x0mu0)=acd69(52)
      brack(ninjaidxt0x1mu0)=0.0_ki
   end subroutine brack_22
!---#] subroutine brack_22:
!---#[ subroutine numerator_t2:
   subroutine numerator_t2(ncut, a0, a1, b, c, param, deg, coeffs) &
   & bind(c, name="p2_gg_httbar_d69h4_qp_ninja_t2")
      use iso_c_binding, only: c_int
      use quadninjago_module, only: ki => ki_nin
      use p2_gg_httbar_globalsl1_qp, only: epspow
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_abbrevd69h4_qp
      implicit none
      integer(c_int), intent(in) :: ncut, deg
      complex(ki), dimension(0:3), intent(in) :: a0, a1, b, c
      complex(ki), dimension(0:2), intent(in) :: param
      complex(ki), dimension(4) :: vecA0, vecA1, vecB, vecC
      complex(ki), dimension(0:*), intent(out) :: coeffs
      integer :: t1
      complex(ki), dimension(4) :: qshift
      qshift = -k2+k5
      vecA0(1:4) = - a0(0:3) - qshift(1:4)
      vecA1(1:4) = - a1(0:3)
      vecB(1:4) = - b(0:3)
      vecC(1:4) = - c(0:3)
      if (deg.lt.0) return
      t1 = 0
      call cond_t(epspow.eq.t1,brack_21,vecA0,vecA1,vecB,vecC,param,coeffs)
      if (deg.le.(1+(-1))) return
      call cond_t(epspow.eq.t1,brack_22,vecA0,vecA1,vecB,vecC,param,coeffs)
   end subroutine numerator_t2
!---#] subroutine numerator_t2:
end module     p2_gg_httbar_d69h4l132_qp
