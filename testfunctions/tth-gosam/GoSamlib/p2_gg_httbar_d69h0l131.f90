module     p2_gg_httbar_d69h0l131
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity0d69h0l131.f90
   ! generator: buildfortran_tn3.py
   use p2_gg_httbar_config, only: ki
   use p2_gg_httbar_util, only: cond_t, d => metric_tensor
   implicit none
   private
   complex(ki), parameter :: i_ = (0.0_ki, 1.0_ki)
   integer, parameter :: ninjaidxt2mu0 = 0
   integer, parameter :: ninjaidxt1mu0 = 1
   integer, parameter :: ninjaidxt0mu0 = 2
   integer, parameter :: ninjaidxt0mu2 = 3
   public :: numerator_t3
contains
!---#[ subroutine brack_31:
   pure subroutine brack_31(ninjaA, ninjaE3, ninjaE4, ninjaP, brack)
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd69h0
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA, ninjaE3, ninjaE4
      complex(ki), intent(in) :: ninjaP
      complex(ki), dimension(1) :: acd69
      complex(ki), dimension (0:*), intent(inout) :: brack
      brack(ninjaidxt2mu0)=0.0_ki
   end subroutine brack_31
!---#] subroutine brack_31:
!---#[ subroutine brack_32:
   pure subroutine brack_32(ninjaA, ninjaE3, ninjaE4, ninjaP, brack)
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd69h0
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA, ninjaE3, ninjaE4
      complex(ki), intent(in) :: ninjaP
      complex(ki), dimension(124) :: acd69
      complex(ki), dimension (0:*), intent(inout) :: brack
      acd69(1)=dotproduct(ninjaE3,spvae2k2)
      acd69(2)=abb69(14)
      acd69(3)=dotproduct(ninjaE3,spvae2k1)
      acd69(4)=abb69(18)
      acd69(5)=dotproduct(ninjaE3,spvak2e2)
      acd69(6)=abb69(22)
      acd69(7)=dotproduct(ninjaE3,spval5e2)
      acd69(8)=abb69(29)
      acd69(9)=dotproduct(ninjaE3,spvae1e2)
      acd69(10)=abb69(43)
      acd69(11)=dotproduct(ninjaE3,spvak1e2)
      acd69(12)=abb69(35)
      acd69(13)=dotproduct(ninjaE3,spval4e2)
      acd69(14)=abb69(56)
      acd69(15)=dotproduct(ninjaE3,spvae2e1)
      acd69(16)=abb69(42)
      acd69(17)=dotproduct(ninjaE3,spvae2l4)
      acd69(18)=abb69(51)
      acd69(19)=dotproduct(k2,ninjaE3)
      acd69(20)=abb69(11)
      acd69(21)=abb69(30)
      acd69(22)=dotproduct(ninjaA,ninjaE3)
      acd69(23)=dotproduct(ninjaE3,spvak2k1)
      acd69(24)=abb69(12)
      acd69(25)=abb69(24)
      acd69(26)=abb69(25)
      acd69(27)=abb69(26)
      acd69(28)=abb69(28)
      acd69(29)=dotproduct(ninjaE3,spvak2e1)
      acd69(30)=abb69(33)
      acd69(31)=dotproduct(ninjaE3,spvak2l4)
      acd69(32)=abb69(36)
      acd69(33)=dotproduct(ninjaE3,spval3e2)
      acd69(34)=dotproduct(ninjaE3,spvak2l3)
      acd69(35)=dotproduct(ninjaE3,spval5l3)
      acd69(36)=dotproduct(ninjaE3,spvak1l3)
      acd69(37)=dotproduct(ninjaE3,spvae1l3)
      acd69(38)=dotproduct(ninjaE3,spval4l3)
      acd69(39)=dotproduct(ninjaE3,spval3k2)
      acd69(40)=dotproduct(ninjaE3,spvae2l3)
      acd69(41)=dotproduct(ninjaE3,spval3k1)
      acd69(42)=dotproduct(ninjaE3,spval3e1)
      acd69(43)=dotproduct(ninjaE3,spval3l4)
      acd69(44)=abb69(21)
      acd69(45)=dotproduct(ninjaE3,spval5k2)
      acd69(46)=abb69(19)
      acd69(47)=dotproduct(ninjaE3,spvak1k2)
      acd69(48)=abb69(20)
      acd69(49)=dotproduct(ninjaE3,spvae1k2)
      acd69(50)=abb69(31)
      acd69(51)=dotproduct(ninjaE3,spval4k2)
      acd69(52)=abb69(32)
      acd69(53)=abb69(37)
      acd69(54)=abb69(52)
      acd69(55)=dotproduct(k2,ninjaA)
      acd69(56)=dotproduct(ninjaA,spvae2k2)
      acd69(57)=dotproduct(ninjaA,spvak2e2)
      acd69(58)=dotproduct(ninjaA,ninjaA)
      acd69(59)=dotproduct(ninjaA,spvae2k1)
      acd69(60)=dotproduct(ninjaA,spval5e2)
      acd69(61)=dotproduct(ninjaA,spvae1e2)
      acd69(62)=dotproduct(ninjaA,spvak1e2)
      acd69(63)=dotproduct(ninjaA,spval4e2)
      acd69(64)=dotproduct(ninjaA,spvae2e1)
      acd69(65)=dotproduct(ninjaA,spvae2l4)
      acd69(66)=abb69(27)
      acd69(67)=dotproduct(ninjaA,spval3e2)
      acd69(68)=dotproduct(ninjaA,spvak2k1)
      acd69(69)=dotproduct(ninjaA,spval3k2)
      acd69(70)=dotproduct(ninjaA,spvae2l3)
      acd69(71)=dotproduct(ninjaA,spval3k1)
      acd69(72)=dotproduct(ninjaA,spval5k2)
      acd69(73)=dotproduct(ninjaA,spvak1k2)
      acd69(74)=dotproduct(ninjaA,spvak2l3)
      acd69(75)=dotproduct(ninjaA,spval5l3)
      acd69(76)=dotproduct(ninjaA,spvae1k2)
      acd69(77)=dotproduct(ninjaA,spval4k2)
      acd69(78)=dotproduct(ninjaA,spvak2e1)
      acd69(79)=dotproduct(ninjaA,spvak1l3)
      acd69(80)=dotproduct(ninjaA,spvak2l4)
      acd69(81)=dotproduct(ninjaA,spval3e1)
      acd69(82)=dotproduct(ninjaA,spvae1l3)
      acd69(83)=dotproduct(ninjaA,spval3l4)
      acd69(84)=dotproduct(ninjaA,spval4l3)
      acd69(85)=abb69(9)
      acd69(86)=abb69(10)
      acd69(87)=abb69(16)
      acd69(88)=abb69(15)
      acd69(89)=abb69(17)
      acd69(90)=abb69(23)
      acd69(91)=abb69(44)
      acd69(92)=abb69(34)
      acd69(93)=abb69(55)
      acd69(94)=abb69(40)
      acd69(95)=abb69(49)
      acd69(96)=acd69(6)*acd69(5)
      acd69(97)=acd69(8)*acd69(7)
      acd69(98)=acd69(3)*acd69(4)
      acd69(99)=acd69(9)*acd69(10)
      acd69(100)=acd69(11)*acd69(12)
      acd69(101)=acd69(13)*acd69(14)
      acd69(102)=acd69(15)*acd69(16)
      acd69(103)=acd69(17)*acd69(18)
      acd69(96)=-acd69(103)-acd69(100)+acd69(98)-acd69(99)+acd69(101)-acd69(102&
      &)+acd69(96)-acd69(97)
      acd69(97)=acd69(2)*acd69(1)
      acd69(97)=-acd69(97)-acd69(96)
      acd69(98)=acd69(22)*acd69(97)
      acd69(99)=acd69(20)*acd69(19)
      acd69(100)=acd69(26)*acd69(9)
      acd69(101)=acd69(27)*acd69(11)
      acd69(102)=acd69(28)*acd69(13)
      acd69(103)=acd69(23)*acd69(24)
      acd69(104)=acd69(29)*acd69(30)
      acd69(105)=acd69(31)*acd69(32)
      acd69(99)=acd69(99)+acd69(103)+acd69(104)+acd69(100)+acd69(101)+acd69(102&
      &)+acd69(105)
      acd69(100)=acd69(25)*acd69(5)
      acd69(100)=acd69(100)+acd69(99)
      acd69(100)=acd69(1)*acd69(100)
      acd69(101)=acd69(21)*acd69(19)
      acd69(102)=acd69(45)*acd69(46)
      acd69(103)=acd69(47)*acd69(48)
      acd69(104)=acd69(49)*acd69(50)
      acd69(105)=acd69(51)*acd69(52)
      acd69(101)=acd69(101)+acd69(102)+acd69(103)+acd69(104)+acd69(105)
      acd69(102)=acd69(5)*acd69(101)
      acd69(103)=acd69(34)*acd69(6)
      acd69(104)=acd69(35)*acd69(8)
      acd69(105)=acd69(36)*acd69(12)
      acd69(106)=acd69(37)*acd69(10)
      acd69(107)=acd69(38)*acd69(14)
      acd69(103)=-acd69(107)-acd69(103)+acd69(104)+acd69(105)+acd69(106)
      acd69(104)=-acd69(33)*acd69(103)
      acd69(105)=acd69(39)*acd69(2)
      acd69(106)=acd69(41)*acd69(4)
      acd69(107)=acd69(42)*acd69(16)
      acd69(108)=acd69(43)*acd69(18)
      acd69(105)=-acd69(105)-acd69(106)+acd69(107)+acd69(108)
      acd69(106)=-acd69(40)*acd69(105)
      acd69(107)=acd69(44)*acd69(3)
      acd69(108)=acd69(53)*acd69(15)
      acd69(109)=acd69(54)*acd69(17)
      acd69(107)=acd69(107)+acd69(108)+acd69(109)
      acd69(108)=acd69(7)*acd69(107)
      acd69(98)=2.0_ki*acd69(98)+acd69(100)+acd69(104)+acd69(102)+acd69(106)+ac&
      &d69(108)
      acd69(100)=ninjaP+acd69(58)
      acd69(96)=-acd69(100)*acd69(96)
      acd69(99)=acd69(56)*acd69(99)
      acd69(102)=-acd69(67)*acd69(103)
      acd69(103)=acd69(74)*acd69(6)
      acd69(104)=-acd69(75)*acd69(8)
      acd69(106)=-acd69(79)*acd69(12)
      acd69(108)=-acd69(82)*acd69(10)
      acd69(109)=acd69(84)*acd69(14)
      acd69(103)=acd69(86)+acd69(109)+acd69(108)+acd69(106)+acd69(104)+acd69(10&
      &3)
      acd69(103)=acd69(33)*acd69(103)
      acd69(104)=2.0_ki*acd69(22)
      acd69(106)=-acd69(6)*acd69(104)
      acd69(101)=acd69(106)+acd69(101)
      acd69(101)=acd69(57)*acd69(101)
      acd69(105)=-acd69(70)*acd69(105)
      acd69(106)=acd69(72)*acd69(46)
      acd69(108)=acd69(73)*acd69(48)
      acd69(109)=acd69(76)*acd69(50)
      acd69(110)=acd69(77)*acd69(52)
      acd69(106)=acd69(89)+acd69(110)+acd69(109)+acd69(108)+acd69(106)
      acd69(106)=acd69(5)*acd69(106)
      acd69(108)=acd69(69)*acd69(2)
      acd69(109)=acd69(71)*acd69(4)
      acd69(110)=-acd69(81)*acd69(16)
      acd69(111)=-acd69(83)*acd69(18)
      acd69(108)=acd69(87)+acd69(111)+acd69(110)+acd69(109)+acd69(108)
      acd69(108)=acd69(40)*acd69(108)
      acd69(109)=acd69(68)*acd69(24)
      acd69(110)=acd69(78)*acd69(30)
      acd69(111)=acd69(80)*acd69(32)
      acd69(109)=acd69(85)+acd69(111)+acd69(110)+acd69(109)
      acd69(109)=acd69(1)*acd69(109)
      acd69(110)=acd69(8)*acd69(104)
      acd69(107)=acd69(110)+acd69(107)
      acd69(107)=acd69(60)*acd69(107)
      acd69(100)=-acd69(1)*acd69(100)
      acd69(110)=-acd69(56)*acd69(104)
      acd69(100)=acd69(110)+acd69(100)
      acd69(100)=acd69(2)*acd69(100)
      acd69(110)=acd69(56)*acd69(5)
      acd69(111)=acd69(57)*acd69(1)
      acd69(110)=acd69(110)+acd69(111)
      acd69(110)=acd69(25)*acd69(110)
      acd69(111)=acd69(20)*acd69(1)
      acd69(112)=acd69(21)*acd69(5)
      acd69(111)=acd69(111)+acd69(112)
      acd69(111)=acd69(55)*acd69(111)
      acd69(112)=-acd69(4)*acd69(104)
      acd69(113)=acd69(44)*acd69(7)
      acd69(112)=acd69(112)+acd69(113)
      acd69(112)=acd69(59)*acd69(112)
      acd69(113)=acd69(10)*acd69(104)
      acd69(114)=acd69(26)*acd69(1)
      acd69(113)=acd69(113)+acd69(114)
      acd69(113)=acd69(61)*acd69(113)
      acd69(114)=acd69(12)*acd69(104)
      acd69(115)=acd69(27)*acd69(1)
      acd69(114)=acd69(114)+acd69(115)
      acd69(114)=acd69(62)*acd69(114)
      acd69(115)=-acd69(14)*acd69(104)
      acd69(116)=acd69(28)*acd69(1)
      acd69(115)=acd69(115)+acd69(116)
      acd69(115)=acd69(63)*acd69(115)
      acd69(116)=acd69(16)*acd69(104)
      acd69(117)=acd69(53)*acd69(7)
      acd69(116)=acd69(116)+acd69(117)
      acd69(116)=acd69(64)*acd69(116)
      acd69(117)=acd69(18)*acd69(104)
      acd69(118)=acd69(54)*acd69(7)
      acd69(117)=acd69(117)+acd69(118)
      acd69(117)=acd69(65)*acd69(117)
      acd69(104)=acd69(66)*acd69(104)
      acd69(118)=acd69(88)*acd69(3)
      acd69(119)=acd69(90)*acd69(7)
      acd69(120)=acd69(91)*acd69(9)
      acd69(121)=acd69(92)*acd69(11)
      acd69(122)=acd69(93)*acd69(13)
      acd69(123)=acd69(94)*acd69(15)
      acd69(124)=acd69(95)*acd69(17)
      acd69(96)=acd69(124)+acd69(123)+acd69(122)+acd69(121)+acd69(120)+acd69(11&
      &9)+acd69(118)+acd69(104)+acd69(117)+acd69(116)+acd69(115)+acd69(114)+acd&
      &69(113)+acd69(112)+acd69(111)+acd69(110)+acd69(100)+acd69(99)+acd69(101)&
      &+acd69(103)+acd69(102)+acd69(108)+acd69(106)+acd69(105)+acd69(107)+acd69&
      &(109)+acd69(96)
      brack(ninjaidxt1mu0)=acd69(98)
      brack(ninjaidxt0mu0)=acd69(96)
      brack(ninjaidxt0mu2)=acd69(97)
   end subroutine brack_32
!---#] subroutine brack_32:
!---#[ subroutine numerator_t3:
   subroutine numerator_t3(ncut, a, b, c, param, deg, coeffs) &
   & bind(c, name="p2_gg_httbar_d69h0_ninja_t3")
      use iso_c_binding, only: c_int
      use ninjago_module, only: ki => ki_nin
      use p2_gg_httbar_globalsl1, only: epspow
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_abbrevd69h0
      implicit none
      integer(c_int), intent(in) :: ncut, deg
      complex(ki), dimension(0:3), intent(in) :: a, b, c
      complex(ki), intent(in) :: param
      complex(ki), dimension(4) :: vecA, vecB, vecC
      complex(ki), dimension(0:*), intent(out) :: coeffs
      integer :: t1
      complex(ki), dimension(4) :: qshift
      qshift = -k2+k5
      vecA(1:4) = - a(0:3) - qshift(1:4)
      vecB(1:4) = - b(0:3)
      vecC(1:4) = - c(0:3)
      if (deg.lt.0) return
      t1 = 0
      call cond_t(epspow.eq.t1,brack_31,vecA,vecB,vecC,param,coeffs)
      if (deg.le.(1+(-1))) return
      call cond_t(epspow.eq.t1,brack_32,vecA,vecB,vecC,param,coeffs)
   end subroutine numerator_t3
!---#] subroutine numerator_t3:
end module     p2_gg_httbar_d69h0l131
