module     p2_gg_httbar_d90h4l132
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity4d90h4l132.f90
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
      use p2_gg_httbar_abbrevd90h4
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA0, ninjaA1, ninjaE3, ninja&
      &E4
      complex(ki), dimension(0:2), intent(in) :: ninjaP
      complex(ki), dimension(14) :: acd90
      complex(ki), dimension (0:*), intent(inout) :: brack
      complex(ki) :: ninjaP0, ninjaP1, ninjaP2
      ninjaP0 = ninjaP(0)
      ninjaP1 = ninjaP(1)
      ninjaP2 = ninjaP(2)
      acd90(1)=dotproduct(k2,ninjaE3)
      acd90(2)=dotproduct(e1,ninjaE3)
      acd90(3)=dotproduct(e2,ninjaE3)
      acd90(4)=abb90(46)
      acd90(5)=dotproduct(ninjaE3,spval5l4)
      acd90(6)=abb90(22)
      acd90(7)=dotproduct(ninjaE3,spvak2l3)
      acd90(8)=abb90(32)
      acd90(9)=dotproduct(ninjaE3,spval3l4)
      acd90(10)=abb90(75)
      acd90(11)=acd90(4)*acd90(1)
      acd90(12)=acd90(6)*acd90(5)
      acd90(13)=acd90(8)*acd90(7)
      acd90(14)=acd90(10)*acd90(9)
      acd90(11)=acd90(14)+acd90(13)+acd90(11)+acd90(12)
      acd90(11)=acd90(11)*acd90(3)*acd90(2)
      brack(ninjaidxt1x0mu0)=acd90(11)
   end subroutine brack_21
!---#] subroutine brack_21:
!---#[ subroutine brack_22:
   pure subroutine brack_22(ninjaA0, ninjaA1, ninjaE3, ninjaE4, ninjaP, brack)
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd90h4
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA0, ninjaA1, ninjaE3, ninja&
      &E4
      complex(ki), dimension(0:2), intent(in) :: ninjaP
      complex(ki), dimension(124) :: acd90
      complex(ki), dimension (0:*), intent(inout) :: brack
      complex(ki) :: ninjaP0, ninjaP1, ninjaP2
      ninjaP0 = ninjaP(0)
      ninjaP1 = ninjaP(1)
      ninjaP2 = ninjaP(2)
      acd90(1)=dotproduct(k2,ninjaA1)
      acd90(2)=dotproduct(e1,ninjaE3)
      acd90(3)=dotproduct(e2,ninjaE3)
      acd90(4)=abb90(46)
      acd90(5)=dotproduct(k2,ninjaE3)
      acd90(6)=dotproduct(e1,ninjaA1)
      acd90(7)=dotproduct(e2,ninjaA1)
      acd90(8)=dotproduct(ninjaE3,spval5l4)
      acd90(9)=abb90(22)
      acd90(10)=dotproduct(ninjaE3,spvak2l3)
      acd90(11)=abb90(32)
      acd90(12)=dotproduct(ninjaE3,spval3l4)
      acd90(13)=abb90(75)
      acd90(14)=dotproduct(ninjaA1,spval5l4)
      acd90(15)=dotproduct(ninjaA1,spvak2l3)
      acd90(16)=dotproduct(ninjaA1,spval3l4)
      acd90(17)=dotproduct(k1,ninjaE3)
      acd90(18)=dotproduct(ninjaE3,spvae2e1)
      acd90(19)=abb90(30)
      acd90(20)=dotproduct(ninjaE3,spvae1e2)
      acd90(21)=abb90(44)
      acd90(22)=dotproduct(k2,ninjaA0)
      acd90(23)=abb90(10)
      acd90(24)=dotproduct(e1,ninjaA0)
      acd90(25)=dotproduct(e2,ninjaA0)
      acd90(26)=abb90(28)
      acd90(27)=abb90(24)
      acd90(28)=dotproduct(ninjaA0,ninjaE3)
      acd90(29)=abb90(27)
      acd90(30)=abb90(29)
      acd90(31)=dotproduct(l4,ninjaE3)
      acd90(32)=abb90(19)
      acd90(33)=abb90(39)
      acd90(34)=abb90(62)
      acd90(35)=abb90(41)
      acd90(36)=dotproduct(ninjaA0,spval5l4)
      acd90(37)=dotproduct(ninjaA0,spvak2l3)
      acd90(38)=dotproduct(ninjaA0,spval3l4)
      acd90(39)=abb90(11)
      acd90(40)=abb90(21)
      acd90(41)=dotproduct(ninjaE3,spvak2l4)
      acd90(42)=abb90(31)
      acd90(43)=dotproduct(ninjaE3,spval5k2)
      acd90(44)=abb90(77)
      acd90(45)=dotproduct(ninjaE3,spval3k2)
      acd90(46)=abb90(65)
      acd90(47)=abb90(61)
      acd90(48)=dotproduct(ninjaE3,spval4k2)
      acd90(49)=abb90(86)
      acd90(50)=dotproduct(ninjaE3,spval4l3)
      acd90(51)=abb90(35)
      acd90(52)=dotproduct(ninjaE3,spvak1e2)
      acd90(53)=abb90(40)
      acd90(54)=abb90(73)
      acd90(55)=dotproduct(ninjaE3,spvae2k1)
      acd90(56)=abb90(42)
      acd90(57)=dotproduct(ninjaE3,spval3e2)
      acd90(58)=abb90(63)
      acd90(59)=dotproduct(ninjaE3,spval5e2)
      acd90(60)=abb90(66)
      acd90(61)=dotproduct(ninjaE3,spvae2k2)
      acd90(62)=abb90(67)
      acd90(63)=dotproduct(ninjaE3,spvae2l4)
      acd90(64)=abb90(78)
      acd90(65)=dotproduct(ninjaE3,spvak2e2)
      acd90(66)=abb90(79)
      acd90(67)=dotproduct(ninjaE3,spvae2l3)
      acd90(68)=abb90(80)
      acd90(69)=abb90(14)
      acd90(70)=abb90(23)
      acd90(71)=abb90(71)
      acd90(72)=abb90(76)
      acd90(73)=dotproduct(ninjaE3,spvak2e1)
      acd90(74)=abb90(18)
      acd90(75)=abb90(89)
      acd90(76)=abb90(57)
      acd90(77)=abb90(74)
      acd90(78)=dotproduct(ninjaE3,spvak1e1)
      acd90(79)=abb90(37)
      acd90(80)=abb90(87)
      acd90(81)=dotproduct(ninjaE3,spvae1k1)
      acd90(82)=abb90(51)
      acd90(83)=dotproduct(ninjaE3,spvae1l4)
      acd90(84)=abb90(68)
      acd90(85)=dotproduct(ninjaE3,spval5e1)
      acd90(86)=abb90(69)
      acd90(87)=dotproduct(ninjaE3,spvae1k2)
      acd90(88)=abb90(70)
      acd90(89)=dotproduct(ninjaE3,spvae1l3)
      acd90(90)=abb90(81)
      acd90(91)=dotproduct(ninjaE3,spval3e1)
      acd90(92)=abb90(82)
      acd90(93)=abb90(64)
      acd90(94)=abb90(33)
      acd90(95)=abb90(43)
      acd90(96)=abb90(53)
      acd90(97)=abb90(60)
      acd90(98)=abb90(13)
      acd90(99)=abb90(15)
      acd90(100)=abb90(26)
      acd90(101)=acd90(1)*acd90(4)
      acd90(102)=acd90(14)*acd90(9)
      acd90(103)=acd90(15)*acd90(11)
      acd90(104)=acd90(16)*acd90(13)
      acd90(101)=acd90(104)+acd90(103)+acd90(102)+acd90(101)
      acd90(102)=acd90(3)*acd90(2)
      acd90(101)=acd90(102)*acd90(101)
      acd90(103)=acd90(4)*acd90(5)
      acd90(104)=acd90(9)*acd90(8)
      acd90(105)=acd90(11)*acd90(10)
      acd90(106)=acd90(13)*acd90(12)
      acd90(103)=acd90(106)+acd90(103)+acd90(104)+acd90(105)
      acd90(104)=acd90(103)*acd90(3)
      acd90(105)=acd90(6)*acd90(104)
      acd90(103)=acd90(103)*acd90(2)
      acd90(106)=acd90(7)*acd90(103)
      acd90(101)=acd90(105)+acd90(106)+acd90(101)
      acd90(105)=acd90(26)*acd90(5)
      acd90(106)=acd90(32)*acd90(31)
      acd90(107)=2.0_ki*acd90(28)
      acd90(108)=acd90(40)*acd90(107)
      acd90(109)=acd90(42)*acd90(41)
      acd90(110)=-acd90(44)*acd90(43)
      acd90(111)=acd90(46)*acd90(45)
      acd90(112)=acd90(47)*acd90(8)
      acd90(113)=acd90(49)*acd90(48)
      acd90(114)=acd90(51)*acd90(50)
      acd90(115)=acd90(53)*acd90(52)
      acd90(116)=acd90(54)*acd90(12)
      acd90(117)=acd90(56)*acd90(55)
      acd90(118)=acd90(58)*acd90(57)
      acd90(119)=acd90(60)*acd90(59)
      acd90(120)=acd90(62)*acd90(61)
      acd90(121)=acd90(64)*acd90(63)
      acd90(122)=acd90(66)*acd90(65)
      acd90(123)=acd90(68)*acd90(67)
      acd90(105)=acd90(123)+acd90(122)+acd90(121)+acd90(120)+acd90(119)+acd90(1&
      &18)+acd90(117)+acd90(116)+acd90(115)+acd90(114)+acd90(113)+acd90(112)+ac&
      &d90(111)+acd90(110)+acd90(109)+acd90(108)+acd90(106)+acd90(105)
      acd90(105)=acd90(2)*acd90(105)
      acd90(106)=acd90(27)*acd90(5)
      acd90(108)=-acd90(33)*acd90(31)
      acd90(109)=acd90(69)*acd90(107)
      acd90(110)=acd90(70)*acd90(41)
      acd90(111)=acd90(71)*acd90(43)
      acd90(112)=acd90(72)*acd90(45)
      acd90(113)=acd90(74)*acd90(73)
      acd90(114)=-acd90(75)*acd90(8)
      acd90(115)=acd90(76)*acd90(48)
      acd90(116)=acd90(77)*acd90(50)
      acd90(117)=acd90(79)*acd90(78)
      acd90(118)=-acd90(80)*acd90(12)
      acd90(119)=acd90(82)*acd90(81)
      acd90(120)=acd90(84)*acd90(83)
      acd90(121)=acd90(86)*acd90(85)
      acd90(122)=acd90(88)*acd90(87)
      acd90(123)=acd90(90)*acd90(89)
      acd90(124)=acd90(92)*acd90(91)
      acd90(106)=acd90(124)+acd90(123)+acd90(122)+acd90(121)+acd90(120)+acd90(1&
      &19)+acd90(118)+acd90(117)+acd90(116)+acd90(115)+acd90(114)+acd90(113)+ac&
      &d90(112)+acd90(111)+acd90(110)+acd90(109)+acd90(108)+acd90(106)
      acd90(106)=acd90(3)*acd90(106)
      acd90(108)=-acd90(29)*acd90(5)
      acd90(109)=acd90(93)*acd90(8)
      acd90(110)=acd90(95)*acd90(18)
      acd90(111)=acd90(96)*acd90(12)
      acd90(112)=acd90(97)*acd90(20)
      acd90(108)=acd90(112)+acd90(111)+acd90(110)+acd90(109)+acd90(108)
      acd90(108)=acd90(107)*acd90(108)
      acd90(109)=acd90(22)*acd90(4)
      acd90(110)=acd90(36)*acd90(9)
      acd90(111)=acd90(37)*acd90(11)
      acd90(112)=acd90(38)*acd90(13)
      acd90(109)=acd90(39)+acd90(112)+acd90(111)+acd90(110)+acd90(109)
      acd90(102)=acd90(102)*acd90(109)
      acd90(104)=acd90(24)*acd90(104)
      acd90(103)=acd90(25)*acd90(103)
      acd90(109)=acd90(98)*acd90(43)
      acd90(110)=acd90(99)*acd90(45)
      acd90(111)=acd90(100)*acd90(48)
      acd90(109)=acd90(111)+acd90(110)+acd90(109)
      acd90(109)=acd90(41)*acd90(109)
      acd90(110)=acd90(19)*acd90(18)
      acd90(111)=-acd90(21)*acd90(20)
      acd90(110)=acd90(111)+acd90(110)
      acd90(111)=acd90(17)-acd90(5)
      acd90(110)=acd90(111)*acd90(110)
      acd90(111)=acd90(34)*acd90(8)
      acd90(112)=acd90(35)*acd90(12)
      acd90(111)=acd90(112)+acd90(111)
      acd90(111)=acd90(31)*acd90(111)
      acd90(107)=-acd90(10)*acd90(107)
      acd90(112)=-acd90(50)*acd90(41)
      acd90(107)=acd90(107)+acd90(112)
      acd90(107)=acd90(94)*acd90(107)
      acd90(112)=-acd90(23)*acd90(5)**2
      acd90(113)=-acd90(30)*acd90(10)*acd90(5)
      acd90(102)=acd90(113)+acd90(112)+acd90(107)+acd90(104)+acd90(103)+acd90(1&
      &06)+acd90(105)+acd90(108)+acd90(102)+acd90(109)+acd90(111)+acd90(110)
      brack(ninjaidxt0x0mu0)=acd90(102)
      brack(ninjaidxt0x1mu0)=acd90(101)
   end subroutine brack_22
!---#] subroutine brack_22:
!---#[ subroutine numerator_t2:
   subroutine numerator_t2(ncut, a0, a1, b, c, param, deg, coeffs) &
   & bind(c, name="p2_gg_httbar_d90h4_ninja_t2")
      use iso_c_binding, only: c_int
      use ninjago_module, only: ki => ki_nin
      use p2_gg_httbar_globalsl1, only: epspow
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_abbrevd90h4
      implicit none
      integer(c_int), intent(in) :: ncut, deg
      complex(ki), dimension(0:3), intent(in) :: a0, a1, b, c
      complex(ki), dimension(0:2), intent(in) :: param
      complex(ki), dimension(4) :: vecA0, vecA1, vecB, vecC
      complex(ki), dimension(0:*), intent(out) :: coeffs
      integer :: t1
      complex(ki), dimension(4) :: qshift
      qshift = -k2
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
end module     p2_gg_httbar_d90h4l132
