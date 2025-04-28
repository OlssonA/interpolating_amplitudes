module     p2_gg_httbar_d12h8l132_qp
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity8d12h8l132_qp.f90
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
      use p2_gg_httbar_abbrevd12h8_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA0, ninjaA1, ninjaE3, ninja&
      &E4
      complex(ki), dimension(0:2), intent(in) :: ninjaP
      complex(ki), dimension(22) :: acd12
      complex(ki), dimension (0:*), intent(inout) :: brack
      complex(ki) :: ninjaP0, ninjaP1, ninjaP2
      ninjaP0 = ninjaP(0)
      ninjaP1 = ninjaP(1)
      ninjaP2 = ninjaP(2)
      acd12(1)=dotproduct(k2,ninjaE3)
      acd12(2)=abb12(10)
      acd12(3)=dotproduct(ninjaE3,spval3k2)
      acd12(4)=abb12(30)
      acd12(5)=dotproduct(ninjaE3,spvak1k2)
      acd12(6)=dotproduct(ninjaE3,spvak2k1)
      acd12(7)=abb12(9)
      acd12(8)=dotproduct(ninjaE3,spval3k1)
      acd12(9)=abb12(23)
      acd12(10)=dotproduct(ninjaE3,spvak1l5)
      acd12(11)=dotproduct(ninjaE3,spval4k1)
      acd12(12)=abb12(19)
      acd12(13)=dotproduct(ninjaE3,spvak1l3)
      acd12(14)=abb12(24)
      acd12(15)=dotproduct(ninjaE3,spvak2l5)
      acd12(16)=dotproduct(ninjaE3,spval4k2)
      acd12(17)=dotproduct(ninjaE3,spvak2l3)
      acd12(18)=acd12(14)*acd12(17)
      acd12(19)=acd12(12)*acd12(15)
      acd12(18)=acd12(18)+acd12(19)
      acd12(18)=acd12(16)*acd12(18)
      acd12(19)=-acd12(14)*acd12(13)
      acd12(20)=-acd12(12)*acd12(10)
      acd12(19)=acd12(19)+acd12(20)
      acd12(19)=acd12(11)*acd12(19)
      acd12(20)=acd12(8)*acd12(9)
      acd12(21)=acd12(6)*acd12(7)
      acd12(20)=acd12(20)+acd12(21)
      acd12(20)=acd12(5)*acd12(20)
      acd12(21)=-acd12(3)*acd12(4)
      acd12(22)=acd12(1)*acd12(2)
      acd12(21)=acd12(21)+acd12(22)
      acd12(21)=acd12(1)*acd12(21)
      acd12(18)=acd12(21)+acd12(20)+acd12(19)+acd12(18)
      brack(ninjaidxt1x0mu0)=acd12(18)
   end subroutine brack_21
!---#] subroutine brack_21:
!---#[ subroutine brack_22:
   pure subroutine brack_22(ninjaA0, ninjaA1, ninjaE3, ninjaE4, ninjaP, brack)
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_color_qp
      use p2_gg_httbar_abbrevd12h8_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA0, ninjaA1, ninjaE3, ninja&
      &E4
      complex(ki), dimension(0:2), intent(in) :: ninjaP
      complex(ki), dimension(73) :: acd12
      complex(ki), dimension (0:*), intent(inout) :: brack
      complex(ki) :: ninjaP0, ninjaP1, ninjaP2
      ninjaP0 = ninjaP(0)
      ninjaP1 = ninjaP(1)
      ninjaP2 = ninjaP(2)
      acd12(1)=dotproduct(k2,ninjaA1)
      acd12(2)=dotproduct(k2,ninjaE3)
      acd12(3)=abb12(10)
      acd12(4)=dotproduct(ninjaE3,spval3k2)
      acd12(5)=abb12(30)
      acd12(6)=dotproduct(ninjaA1,spval3k2)
      acd12(7)=dotproduct(ninjaA1,spvak1k2)
      acd12(8)=dotproduct(ninjaE3,spvak2k1)
      acd12(9)=abb12(9)
      acd12(10)=dotproduct(ninjaE3,spval3k1)
      acd12(11)=abb12(23)
      acd12(12)=dotproduct(ninjaA1,spvak2k1)
      acd12(13)=dotproduct(ninjaE3,spvak1k2)
      acd12(14)=dotproduct(ninjaA1,spval3k1)
      acd12(15)=dotproduct(ninjaA1,spval4k2)
      acd12(16)=dotproduct(ninjaE3,spvak2l5)
      acd12(17)=abb12(19)
      acd12(18)=dotproduct(ninjaE3,spvak2l3)
      acd12(19)=abb12(24)
      acd12(20)=dotproduct(ninjaA1,spvak1l3)
      acd12(21)=dotproduct(ninjaE3,spval4k1)
      acd12(22)=dotproduct(ninjaA1,spval4k1)
      acd12(23)=dotproduct(ninjaE3,spvak1l3)
      acd12(24)=dotproduct(ninjaE3,spvak1l5)
      acd12(25)=dotproduct(ninjaA1,spvak1l5)
      acd12(26)=dotproduct(ninjaA1,spvak2l5)
      acd12(27)=dotproduct(ninjaE3,spval4k2)
      acd12(28)=dotproduct(ninjaA1,spvak2l3)
      acd12(29)=dotproduct(k2,ninjaA0)
      acd12(30)=dotproduct(ninjaA0,spval3k2)
      acd12(31)=abb12(11)
      acd12(32)=dotproduct(ninjaA0,spvak1k2)
      acd12(33)=dotproduct(ninjaA0,spvak2k1)
      acd12(34)=dotproduct(ninjaA0,spval3k1)
      acd12(35)=dotproduct(ninjaA0,spval4k2)
      acd12(36)=dotproduct(ninjaA0,spvak1l3)
      acd12(37)=dotproduct(ninjaA0,spval4k1)
      acd12(38)=dotproduct(ninjaA0,spvak1l5)
      acd12(39)=dotproduct(ninjaA0,spvak2l5)
      acd12(40)=dotproduct(ninjaA0,spvak2l3)
      acd12(41)=abb12(13)
      acd12(42)=abb12(17)
      acd12(43)=abb12(12)
      acd12(44)=abb12(14)
      acd12(45)=abb12(15)
      acd12(46)=abb12(16)
      acd12(47)=abb12(18)
      acd12(48)=abb12(27)
      acd12(49)=abb12(22)
      acd12(50)=abb12(21)
      acd12(51)=acd12(4)*acd12(5)
      acd12(52)=acd12(3)*acd12(2)
      acd12(51)=-acd12(51)+2.0_ki*acd12(52)
      acd12(52)=acd12(1)*acd12(51)
      acd12(53)=acd12(8)*acd12(9)
      acd12(54)=acd12(10)*acd12(11)
      acd12(53)=acd12(53)+acd12(54)
      acd12(54)=acd12(7)*acd12(53)
      acd12(55)=acd12(16)*acd12(17)
      acd12(56)=acd12(18)*acd12(19)
      acd12(55)=acd12(55)+acd12(56)
      acd12(56)=acd12(15)*acd12(55)
      acd12(57)=acd12(23)*acd12(19)
      acd12(58)=acd12(24)*acd12(17)
      acd12(57)=acd12(57)+acd12(58)
      acd12(58)=-acd12(22)*acd12(57)
      acd12(59)=acd12(5)*acd12(2)
      acd12(60)=-acd12(6)*acd12(59)
      acd12(61)=acd12(9)*acd12(13)
      acd12(62)=acd12(12)*acd12(61)
      acd12(63)=acd12(11)*acd12(13)
      acd12(64)=acd12(14)*acd12(63)
      acd12(65)=acd12(21)*acd12(19)
      acd12(66)=-acd12(20)*acd12(65)
      acd12(67)=acd12(21)*acd12(17)
      acd12(68)=-acd12(25)*acd12(67)
      acd12(69)=acd12(27)*acd12(17)
      acd12(70)=acd12(26)*acd12(69)
      acd12(71)=acd12(27)*acd12(19)
      acd12(72)=acd12(28)*acd12(71)
      acd12(52)=acd12(72)+acd12(70)+acd12(68)+acd12(66)+acd12(64)+acd12(62)+acd&
      &12(60)+acd12(58)+acd12(56)+acd12(54)+acd12(52)
      acd12(51)=acd12(29)*acd12(51)
      acd12(53)=acd12(32)*acd12(53)
      acd12(54)=acd12(35)*acd12(55)
      acd12(55)=-acd12(37)*acd12(57)
      acd12(56)=-acd12(30)*acd12(59)
      acd12(57)=acd12(31)*acd12(2)
      acd12(58)=acd12(33)*acd12(61)
      acd12(59)=acd12(34)*acd12(63)
      acd12(60)=-acd12(36)*acd12(65)
      acd12(61)=-acd12(38)*acd12(67)
      acd12(62)=acd12(39)*acd12(69)
      acd12(63)=acd12(40)*acd12(71)
      acd12(64)=acd12(41)*acd12(13)
      acd12(65)=acd12(42)*acd12(8)
      acd12(66)=acd12(43)*acd12(10)
      acd12(67)=acd12(44)*acd12(27)
      acd12(68)=acd12(45)*acd12(23)
      acd12(69)=acd12(46)*acd12(4)
      acd12(70)=acd12(47)*acd12(21)
      acd12(71)=acd12(48)*acd12(24)
      acd12(72)=acd12(49)*acd12(16)
      acd12(73)=acd12(50)*acd12(18)
      acd12(51)=acd12(73)+acd12(72)+acd12(71)+acd12(70)+acd12(69)+acd12(68)+acd&
      &12(67)+acd12(66)+acd12(65)+acd12(64)+acd12(63)+acd12(62)+acd12(61)+acd12&
      &(60)+acd12(59)+acd12(58)+acd12(57)+acd12(56)+acd12(55)+acd12(54)+acd12(5&
      &1)+acd12(53)
      brack(ninjaidxt0x0mu0)=acd12(51)
      brack(ninjaidxt0x1mu0)=acd12(52)
   end subroutine brack_22
!---#] subroutine brack_22:
!---#[ subroutine numerator_t2:
   subroutine numerator_t2(ncut, a0, a1, b, c, param, deg, coeffs) &
   & bind(c, name="p2_gg_httbar_d12h8_qp_ninja_t2")
      use iso_c_binding, only: c_int
      use quadninjago_module, only: ki => ki_nin
      use p2_gg_httbar_globalsl1_qp, only: epspow
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_abbrevd12h8_qp
      implicit none
      integer(c_int), intent(in) :: ncut, deg
      complex(ki), dimension(0:3), intent(in) :: a0, a1, b, c
      complex(ki), dimension(0:2), intent(in) :: param
      complex(ki), dimension(4) :: vecA0, vecA1, vecB, vecC
      complex(ki), dimension(0:*), intent(out) :: coeffs
      integer :: t1
      complex(ki), dimension(4) :: qshift
      qshift = k4
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
end module     p2_gg_httbar_d12h8l132_qp
