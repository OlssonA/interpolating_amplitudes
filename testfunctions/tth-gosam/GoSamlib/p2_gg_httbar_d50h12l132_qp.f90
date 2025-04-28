module     p2_gg_httbar_d50h12l132_qp
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity12d50h12l132_qp.f90
   ! generator: buildfortran_tn3.py
   use p2_gg_httbar_config, only: ki => ki_qp
   use p2_gg_httbar_util_qp, only: cond_t, d => metric_tensor
   implicit none
   private
   complex(ki), parameter :: i_ = (0.0_ki, 1.0_ki)
   integer, parameter :: ninjaidxt2x0mu0 = 0
   integer, parameter :: ninjaidxt1x0mu0 = 1
   integer, parameter :: ninjaidxt1x1mu0 = 2
   integer, parameter :: ninjaidxt0x0mu0 = 3
   integer, parameter :: ninjaidxt0x0mu2 = 4
   integer, parameter :: ninjaidxt0x1mu0 = 5
   integer, parameter :: ninjaidxt0x2mu0 = 6
   public :: numerator_t2
contains
!---#[ subroutine brack_21:
   pure subroutine brack_21(ninjaA0, ninjaA1, ninjaE3, ninjaE4, ninjaP, brack)
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_color_qp
      use p2_gg_httbar_abbrevd50h12_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA0, ninjaA1, ninjaE3, ninja&
      &E4
      complex(ki), dimension(0:2), intent(in) :: ninjaP
      complex(ki), dimension(76) :: acd50
      complex(ki), dimension (0:*), intent(inout) :: brack
      complex(ki) :: ninjaP0, ninjaP1, ninjaP2
      ninjaP0 = ninjaP(0)
      ninjaP1 = ninjaP(1)
      ninjaP2 = ninjaP(2)
      acd50(1)=dotproduct(k2,ninjaA0)
      acd50(2)=dotproduct(ninjaE3,spvak2l5)
      acd50(3)=abb50(35)
      acd50(4)=dotproduct(ninjaE3,spvak2l4)
      acd50(5)=abb50(32)
      acd50(6)=dotproduct(ninjaE3,spvak2l3)
      acd50(7)=abb50(33)
      acd50(8)=dotproduct(k2,ninjaE3)
      acd50(9)=dotproduct(ninjaA0,spvak2l5)
      acd50(10)=dotproduct(ninjaA0,spvak2l4)
      acd50(11)=dotproduct(ninjaA0,spvak2l3)
      acd50(12)=abb50(13)
      acd50(13)=dotproduct(ninjaA0,spvak1l4)
      acd50(14)=dotproduct(ninjaE3,spvak2k1)
      acd50(15)=abb50(15)
      acd50(16)=dotproduct(ninjaE3,spval3k1)
      acd50(17)=abb50(31)
      acd50(18)=dotproduct(ninjaA0,spvak1l5)
      acd50(19)=abb50(10)
      acd50(20)=abb50(26)
      acd50(21)=dotproduct(ninjaA0,spvak2k1)
      acd50(22)=dotproduct(ninjaE3,spvak1l4)
      acd50(23)=dotproduct(ninjaE3,spvak1l5)
      acd50(24)=dotproduct(ninjaE3,spvak1l3)
      acd50(25)=abb50(14)
      acd50(26)=dotproduct(ninjaA0,spvak1l3)
      acd50(27)=dotproduct(ninjaA0,spval3k2)
      acd50(28)=dotproduct(ninjaA0,spval3k1)
      acd50(29)=dotproduct(ninjaE3,spval3k2)
      acd50(30)=abb50(9)
      acd50(31)=abb50(16)
      acd50(32)=abb50(11)
      acd50(33)=abb50(18)
      acd50(34)=abb50(23)
      acd50(35)=abb50(30)
      acd50(36)=abb50(27)
      acd50(37)=abb50(28)
      acd50(38)=dotproduct(k2,ninjaA1)
      acd50(39)=dotproduct(ninjaA1,spvak2l5)
      acd50(40)=dotproduct(ninjaA1,spvak2l4)
      acd50(41)=dotproduct(ninjaA1,spvak2l3)
      acd50(42)=dotproduct(ninjaA1,spvak1l4)
      acd50(43)=dotproduct(ninjaA1,spvak1l5)
      acd50(44)=dotproduct(ninjaA1,spvak2k1)
      acd50(45)=dotproduct(ninjaA1,spvak1l3)
      acd50(46)=dotproduct(ninjaA1,spval3k2)
      acd50(47)=dotproduct(ninjaA1,spval3k1)
      acd50(48)=acd50(15)*acd50(22)
      acd50(49)=acd50(19)*acd50(23)
      acd50(50)=acd50(24)*acd50(25)
      acd50(48)=acd50(50)+acd50(48)+acd50(49)
      acd50(49)=acd50(21)*acd50(48)
      acd50(50)=acd50(3)*acd50(2)
      acd50(51)=acd50(5)*acd50(4)
      acd50(52)=acd50(6)*acd50(7)
      acd50(50)=acd50(50)+acd50(51)+acd50(52)
      acd50(51)=acd50(1)*acd50(50)
      acd50(52)=acd50(29)*acd50(20)
      acd50(53)=acd50(3)*acd50(8)
      acd50(52)=acd50(52)+acd50(53)
      acd50(53)=acd50(9)*acd50(52)
      acd50(54)=acd50(29)*acd50(17)
      acd50(55)=acd50(5)*acd50(8)
      acd50(54)=acd50(54)+acd50(55)
      acd50(55)=acd50(10)*acd50(54)
      acd50(56)=acd50(16)*acd50(17)
      acd50(57)=acd50(15)*acd50(14)
      acd50(56)=acd50(56)-acd50(57)
      acd50(57)=-acd50(13)*acd50(56)
      acd50(58)=acd50(16)*acd50(20)
      acd50(59)=acd50(19)*acd50(14)
      acd50(58)=acd50(58)-acd50(59)
      acd50(59)=-acd50(18)*acd50(58)
      acd50(60)=acd50(2)*acd50(20)
      acd50(61)=acd50(4)*acd50(17)
      acd50(60)=acd50(60)+acd50(61)
      acd50(61)=acd50(27)*acd50(60)
      acd50(62)=acd50(22)*acd50(17)
      acd50(63)=acd50(23)*acd50(20)
      acd50(62)=acd50(62)+acd50(63)
      acd50(63)=-acd50(28)*acd50(62)
      acd50(64)=acd50(7)*acd50(8)
      acd50(65)=acd50(11)*acd50(64)
      acd50(66)=acd50(12)*acd50(8)
      acd50(67)=acd50(25)*acd50(14)
      acd50(68)=acd50(26)*acd50(67)
      acd50(69)=acd50(30)*acd50(22)
      acd50(70)=acd50(31)*acd50(23)
      acd50(71)=acd50(32)*acd50(14)
      acd50(72)=acd50(33)*acd50(29)
      acd50(73)=acd50(34)*acd50(16)
      acd50(74)=acd50(35)*acd50(2)
      acd50(75)=acd50(36)*acd50(4)
      acd50(76)=acd50(37)*acd50(6)
      acd50(49)=acd50(76)+acd50(75)+acd50(74)+acd50(73)+acd50(72)+acd50(71)+acd&
      &50(70)+acd50(69)+acd50(68)+acd50(66)+acd50(65)+acd50(63)+acd50(61)+acd50&
      &(59)+acd50(57)+acd50(55)+acd50(53)+acd50(51)+acd50(49)
      acd50(51)=acd50(38)*acd50(50)
      acd50(53)=acd50(44)*acd50(48)
      acd50(52)=acd50(39)*acd50(52)
      acd50(54)=acd50(40)*acd50(54)
      acd50(55)=-acd50(42)*acd50(56)
      acd50(56)=-acd50(43)*acd50(58)
      acd50(57)=acd50(46)*acd50(60)
      acd50(58)=-acd50(47)*acd50(62)
      acd50(59)=acd50(41)*acd50(64)
      acd50(61)=acd50(45)*acd50(67)
      acd50(51)=acd50(61)+acd50(59)+acd50(58)+acd50(57)+acd50(56)+acd50(55)+acd&
      &50(54)+acd50(52)+acd50(51)+acd50(53)
      acd50(50)=acd50(8)*acd50(50)
      acd50(48)=acd50(14)*acd50(48)
      acd50(52)=-acd50(16)*acd50(62)
      acd50(53)=acd50(29)*acd50(60)
      acd50(48)=acd50(53)+acd50(48)+acd50(50)+acd50(52)
      brack(ninjaidxt2x0mu0)=acd50(48)
      brack(ninjaidxt1x0mu0)=acd50(49)
      brack(ninjaidxt1x1mu0)=acd50(51)
   end subroutine brack_21
!---#] subroutine brack_21:
!---#[ subroutine brack_22:
   pure subroutine brack_22(ninjaA0, ninjaA1, ninjaE3, ninjaE4, ninjaP, brack)
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_color_qp
      use p2_gg_httbar_abbrevd50h12_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA0, ninjaA1, ninjaE3, ninja&
      &E4
      complex(ki), dimension(0:2), intent(in) :: ninjaP
      complex(ki), dimension(82) :: acd50
      complex(ki), dimension (0:*), intent(inout) :: brack
      complex(ki) :: ninjaP0, ninjaP1, ninjaP2
      ninjaP0 = ninjaP(0)
      ninjaP1 = ninjaP(1)
      ninjaP2 = ninjaP(2)
      acd50(1)=dotproduct(k2,ninjaE3)
      acd50(2)=dotproduct(ninjaE4,spvak2l5)
      acd50(3)=abb50(35)
      acd50(4)=dotproduct(ninjaE4,spvak2l4)
      acd50(5)=abb50(32)
      acd50(6)=dotproduct(ninjaE4,spvak2l3)
      acd50(7)=abb50(33)
      acd50(8)=dotproduct(k2,ninjaE4)
      acd50(9)=dotproduct(ninjaE3,spvak2l5)
      acd50(10)=dotproduct(ninjaE3,spvak2l4)
      acd50(11)=dotproduct(ninjaE3,spvak2l3)
      acd50(12)=dotproduct(ninjaE3,spvak1l4)
      acd50(13)=dotproduct(ninjaE4,spvak2k1)
      acd50(14)=abb50(15)
      acd50(15)=dotproduct(ninjaE4,spval3k1)
      acd50(16)=abb50(31)
      acd50(17)=dotproduct(ninjaE3,spvak1l5)
      acd50(18)=abb50(10)
      acd50(19)=abb50(26)
      acd50(20)=dotproduct(ninjaE3,spvak2k1)
      acd50(21)=dotproduct(ninjaE4,spvak1l4)
      acd50(22)=dotproduct(ninjaE4,spvak1l5)
      acd50(23)=dotproduct(ninjaE4,spvak1l3)
      acd50(24)=abb50(14)
      acd50(25)=dotproduct(ninjaE3,spvak1l3)
      acd50(26)=dotproduct(ninjaE3,spval3k2)
      acd50(27)=dotproduct(ninjaE3,spval3k1)
      acd50(28)=dotproduct(ninjaE4,spval3k2)
      acd50(29)=dotproduct(k2,ninjaA0)
      acd50(30)=dotproduct(ninjaA1,spvak2l5)
      acd50(31)=dotproduct(ninjaA1,spvak2l4)
      acd50(32)=dotproduct(ninjaA1,spvak2l3)
      acd50(33)=dotproduct(k2,ninjaA1)
      acd50(34)=dotproduct(ninjaA0,spvak2l5)
      acd50(35)=dotproduct(ninjaA0,spvak2l4)
      acd50(36)=dotproduct(ninjaA0,spvak2l3)
      acd50(37)=abb50(13)
      acd50(38)=dotproduct(ninjaA0,spvak1l4)
      acd50(39)=dotproduct(ninjaA1,spvak2k1)
      acd50(40)=dotproduct(ninjaA1,spval3k1)
      acd50(41)=dotproduct(ninjaA0,spvak1l5)
      acd50(42)=dotproduct(ninjaA0,spvak2k1)
      acd50(43)=dotproduct(ninjaA1,spvak1l4)
      acd50(44)=dotproduct(ninjaA1,spvak1l5)
      acd50(45)=dotproduct(ninjaA1,spvak1l3)
      acd50(46)=dotproduct(ninjaA0,spvak1l3)
      acd50(47)=dotproduct(ninjaA0,spval3k2)
      acd50(48)=dotproduct(ninjaA0,spval3k1)
      acd50(49)=dotproduct(ninjaA1,spval3k2)
      acd50(50)=abb50(9)
      acd50(51)=abb50(16)
      acd50(52)=abb50(11)
      acd50(53)=abb50(18)
      acd50(54)=abb50(23)
      acd50(55)=abb50(30)
      acd50(56)=abb50(27)
      acd50(57)=abb50(28)
      acd50(58)=abb50(12)
      acd50(59)=acd50(4)*acd50(5)
      acd50(60)=acd50(2)*acd50(3)
      acd50(61)=acd50(6)*acd50(7)
      acd50(59)=acd50(60)+acd50(59)+acd50(61)
      acd50(59)=acd50(1)*acd50(59)
      acd50(60)=acd50(5)*acd50(10)
      acd50(61)=acd50(3)*acd50(9)
      acd50(62)=acd50(7)*acd50(11)
      acd50(60)=acd50(61)+acd50(60)+acd50(62)
      acd50(60)=acd50(8)*acd50(60)
      acd50(61)=acd50(14)*acd50(21)
      acd50(62)=acd50(23)*acd50(24)
      acd50(63)=acd50(18)*acd50(22)
      acd50(61)=acd50(61)+acd50(62)+acd50(63)
      acd50(61)=acd50(20)*acd50(61)
      acd50(62)=acd50(13)*acd50(14)
      acd50(63)=acd50(16)*acd50(15)
      acd50(62)=-acd50(63)+acd50(62)
      acd50(62)=acd50(12)*acd50(62)
      acd50(63)=acd50(18)*acd50(13)
      acd50(64)=acd50(15)*acd50(19)
      acd50(63)=acd50(63)-acd50(64)
      acd50(63)=acd50(63)*acd50(17)
      acd50(64)=acd50(2)*acd50(19)
      acd50(65)=acd50(4)*acd50(16)
      acd50(64)=acd50(64)+acd50(65)
      acd50(64)=acd50(64)*acd50(26)
      acd50(65)=acd50(9)*acd50(19)
      acd50(66)=acd50(10)*acd50(16)
      acd50(65)=acd50(65)+acd50(66)
      acd50(65)=acd50(65)*acd50(28)
      acd50(66)=acd50(21)*acd50(16)
      acd50(67)=acd50(22)*acd50(19)
      acd50(66)=acd50(66)+acd50(67)
      acd50(66)=acd50(66)*acd50(27)
      acd50(67)=acd50(13)*acd50(25)*acd50(24)
      acd50(59)=acd50(67)+acd50(61)+acd50(60)+acd50(59)+acd50(62)+acd50(63)+acd&
      &50(64)+acd50(65)-acd50(66)
      acd50(60)=ninjaP1*acd50(59)
      acd50(61)=acd50(30)*acd50(3)
      acd50(62)=acd50(31)*acd50(5)
      acd50(63)=acd50(32)*acd50(7)
      acd50(61)=acd50(63)+acd50(61)+acd50(62)
      acd50(62)=acd50(29)*acd50(61)
      acd50(63)=acd50(45)*acd50(24)
      acd50(64)=acd50(43)*acd50(14)
      acd50(65)=acd50(44)*acd50(18)
      acd50(64)=acd50(63)+acd50(65)+acd50(64)
      acd50(64)=acd50(42)*acd50(64)
      acd50(65)=acd50(34)*acd50(3)
      acd50(66)=acd50(35)*acd50(5)
      acd50(67)=acd50(36)*acd50(7)
      acd50(65)=acd50(37)+acd50(65)+acd50(66)+acd50(67)
      acd50(66)=acd50(33)*acd50(65)
      acd50(67)=acd50(34)*acd50(19)
      acd50(68)=acd50(35)*acd50(16)
      acd50(67)=acd50(53)+acd50(67)+acd50(68)
      acd50(68)=acd50(49)*acd50(67)
      acd50(69)=acd50(38)*acd50(14)
      acd50(70)=acd50(46)*acd50(24)
      acd50(69)=acd50(52)+acd50(69)+acd50(70)
      acd50(70)=acd50(39)*acd50(69)
      acd50(71)=acd50(39)*acd50(18)
      acd50(72)=acd50(40)*acd50(19)
      acd50(71)=acd50(71)-acd50(72)
      acd50(72)=acd50(41)*acd50(71)
      acd50(73)=acd50(30)*acd50(19)
      acd50(74)=acd50(31)*acd50(16)
      acd50(73)=acd50(73)+acd50(74)
      acd50(74)=acd50(47)*acd50(73)
      acd50(75)=-acd50(43)*acd50(16)
      acd50(76)=-acd50(44)*acd50(19)
      acd50(75)=acd50(75)+acd50(76)
      acd50(75)=acd50(48)*acd50(75)
      acd50(76)=acd50(38)*acd50(16)
      acd50(76)=acd50(76)-acd50(54)
      acd50(77)=-acd50(40)*acd50(76)
      acd50(78)=acd50(50)*acd50(43)
      acd50(79)=acd50(51)*acd50(44)
      acd50(80)=acd50(55)*acd50(30)
      acd50(81)=acd50(56)*acd50(31)
      acd50(82)=acd50(57)*acd50(32)
      acd50(60)=acd50(82)+acd50(81)+acd50(80)+acd50(79)+acd50(78)+acd50(75)+acd&
      &50(74)+acd50(72)+acd50(66)+acd50(64)+acd50(70)+acd50(62)+acd50(68)+acd50&
      &(77)+acd50(60)
      acd50(62)=ninjaP2*acd50(59)
      acd50(61)=acd50(33)*acd50(61)
      acd50(64)=acd50(39)*acd50(14)
      acd50(66)=-acd50(40)*acd50(16)
      acd50(64)=acd50(64)+acd50(66)
      acd50(64)=acd50(43)*acd50(64)
      acd50(66)=acd50(44)*acd50(71)
      acd50(68)=acd50(49)*acd50(73)
      acd50(63)=acd50(39)*acd50(63)
      acd50(61)=acd50(63)+acd50(68)+acd50(66)+acd50(64)+acd50(61)+acd50(62)
      acd50(62)=ninjaP0*acd50(59)
      acd50(63)=acd50(29)*acd50(65)
      acd50(64)=acd50(47)*acd50(67)
      acd50(65)=-acd50(41)*acd50(19)
      acd50(65)=acd50(65)-acd50(76)
      acd50(65)=acd50(48)*acd50(65)
      acd50(66)=acd50(41)*acd50(18)
      acd50(66)=acd50(66)+acd50(69)
      acd50(66)=acd50(42)*acd50(66)
      acd50(67)=acd50(50)*acd50(38)
      acd50(68)=acd50(51)*acd50(41)
      acd50(69)=acd50(55)*acd50(34)
      acd50(70)=acd50(56)*acd50(35)
      acd50(71)=acd50(57)*acd50(36)
      acd50(62)=acd50(58)+acd50(71)+acd50(70)+acd50(69)+acd50(68)+acd50(67)+acd&
      &50(66)+acd50(63)+acd50(65)+acd50(64)+acd50(62)
      brack(ninjaidxt0x0mu0)=acd50(62)
      brack(ninjaidxt0x0mu2)=acd50(59)
      brack(ninjaidxt0x1mu0)=acd50(60)
      brack(ninjaidxt0x2mu0)=acd50(61)
   end subroutine brack_22
!---#] subroutine brack_22:
!---#[ subroutine numerator_t2:
   subroutine numerator_t2(ncut, a0, a1, b, c, param, deg, coeffs) &
   & bind(c, name="p2_gg_httbar_d50h12_qp_ninja_t2")
      use iso_c_binding, only: c_int
      use quadninjago_module, only: ki => ki_nin
      use p2_gg_httbar_globalsl1_qp, only: epspow
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_abbrevd50h12_qp
      implicit none
      integer(c_int), intent(in) :: ncut, deg
      complex(ki), dimension(0:3), intent(in) :: a0, a1, b, c
      complex(ki), dimension(0:2), intent(in) :: param
      complex(ki), dimension(4) :: vecA0, vecA1, vecB, vecC
      complex(ki), dimension(0:*), intent(out) :: coeffs
      integer :: t1
      complex(ki), dimension(4) :: qshift
      qshift = -k3-k5
      vecA0(1:4) = + a0(0:3) - qshift(1:4)
      vecA1(1:4) = + a1(0:3)
      vecB(1:4) = + b(0:3)
      vecC(1:4) = + c(0:3)
      if (deg.lt.0) return
      t1 = 0
      call cond_t(epspow.eq.t1,brack_21,vecA0,vecA1,vecB,vecC,param,coeffs)
      if (deg.le.(1+(0))) return
      call cond_t(epspow.eq.t1,brack_22,vecA0,vecA1,vecB,vecC,param,coeffs)
   end subroutine numerator_t2
!---#] subroutine numerator_t2:
end module     p2_gg_httbar_d50h12l132_qp
