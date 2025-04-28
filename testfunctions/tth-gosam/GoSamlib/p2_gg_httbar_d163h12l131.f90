module     p2_gg_httbar_d163h12l131
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity12d163h12l131.f90
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
      use p2_gg_httbar_abbrevd163h12
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA, ninjaE3, ninjaE4
      complex(ki), intent(in) :: ninjaP
      complex(ki), dimension(28) :: acd163
      complex(ki), dimension (0:*), intent(inout) :: brack
      acd163(1)=dotproduct(k2,ninjaE3)
      acd163(2)=dotproduct(e1,ninjaE3)
      acd163(3)=abb163(62)
      acd163(4)=dotproduct(l5,ninjaE3)
      acd163(5)=abb163(45)
      acd163(6)=dotproduct(ninjaE3,spvak2k1)
      acd163(7)=abb163(17)
      acd163(8)=dotproduct(ninjaE3,spvak2l5)
      acd163(9)=abb163(20)
      acd163(10)=dotproduct(ninjaE3,spvak2l4)
      acd163(11)=abb163(23)
      acd163(12)=dotproduct(ninjaE3,spvak1l5)
      acd163(13)=abb163(26)
      acd163(14)=dotproduct(ninjaE3,spval4l5)
      acd163(15)=abb163(27)
      acd163(16)=dotproduct(ninjaE3,spvae2l5)
      acd163(17)=abb163(70)
      acd163(18)=dotproduct(ninjaE3,spvak2e2)
      acd163(19)=abb163(80)
      acd163(20)=acd163(3)*acd163(1)
      acd163(21)=acd163(5)*acd163(4)
      acd163(22)=acd163(7)*acd163(6)
      acd163(23)=acd163(9)*acd163(8)
      acd163(24)=acd163(11)*acd163(10)
      acd163(25)=acd163(13)*acd163(12)
      acd163(26)=acd163(15)*acd163(14)
      acd163(27)=acd163(17)*acd163(16)
      acd163(28)=acd163(19)*acd163(18)
      acd163(20)=acd163(28)+acd163(27)+acd163(26)+acd163(25)+acd163(24)+acd163(&
      &23)+acd163(22)+acd163(20)+acd163(21)
      acd163(20)=acd163(2)*acd163(20)
      brack(ninjaidxt2mu0)=acd163(20)
   end subroutine brack_31
!---#] subroutine brack_31:
!---#[ subroutine brack_32:
   pure subroutine brack_32(ninjaA, ninjaE3, ninjaE4, ninjaP, brack)
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd163h12
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA, ninjaE3, ninjaE4
      complex(ki), intent(in) :: ninjaP
      complex(ki), dimension(129) :: acd163
      complex(ki), dimension (0:*), intent(inout) :: brack
      acd163(1)=dotproduct(k2,ninjaE3)
      acd163(2)=dotproduct(e1,ninjaE4)
      acd163(3)=abb163(62)
      acd163(4)=dotproduct(k2,ninjaE4)
      acd163(5)=dotproduct(e1,ninjaE3)
      acd163(6)=dotproduct(l5,ninjaE3)
      acd163(7)=abb163(45)
      acd163(8)=dotproduct(l5,ninjaE4)
      acd163(9)=dotproduct(ninjaE4,spvak2l5)
      acd163(10)=abb163(20)
      acd163(11)=dotproduct(ninjaE4,spvak2k1)
      acd163(12)=abb163(17)
      acd163(13)=dotproduct(ninjaE4,spvak2l4)
      acd163(14)=abb163(23)
      acd163(15)=dotproduct(ninjaE4,spvak1l5)
      acd163(16)=abb163(26)
      acd163(17)=dotproduct(ninjaE4,spval4l5)
      acd163(18)=abb163(27)
      acd163(19)=dotproduct(ninjaE4,spvae2l5)
      acd163(20)=abb163(70)
      acd163(21)=dotproduct(ninjaE4,spvak2e2)
      acd163(22)=abb163(80)
      acd163(23)=dotproduct(ninjaE3,spvak2l5)
      acd163(24)=dotproduct(ninjaE3,spvak2k1)
      acd163(25)=dotproduct(ninjaE3,spvak2l4)
      acd163(26)=dotproduct(ninjaE3,spvak1l5)
      acd163(27)=dotproduct(ninjaE3,spval4l5)
      acd163(28)=dotproduct(ninjaE3,spvae2l5)
      acd163(29)=dotproduct(ninjaE3,spvak2e2)
      acd163(30)=abb163(83)
      acd163(31)=dotproduct(k2,ninjaA)
      acd163(32)=dotproduct(e1,ninjaA)
      acd163(33)=abb163(50)
      acd163(34)=dotproduct(l5,ninjaA)
      acd163(35)=abb163(18)
      acd163(36)=dotproduct(ninjaA,spvak2l5)
      acd163(37)=dotproduct(ninjaA,spvak2k1)
      acd163(38)=dotproduct(ninjaA,spvak2l4)
      acd163(39)=dotproduct(ninjaA,spvak1l5)
      acd163(40)=dotproduct(ninjaA,spval4l5)
      acd163(41)=dotproduct(ninjaA,spvae2l5)
      acd163(42)=dotproduct(ninjaA,spvak2e2)
      acd163(43)=abb163(15)
      acd163(44)=dotproduct(ninjaA,ninjaE3)
      acd163(45)=abb163(11)
      acd163(46)=dotproduct(ninjaE3,spvae2k2)
      acd163(47)=abb163(12)
      acd163(48)=dotproduct(ninjaE3,spvae2e1)
      acd163(49)=abb163(14)
      acd163(50)=dotproduct(ninjaE3,spvak2e1)
      acd163(51)=abb163(16)
      acd163(52)=dotproduct(ninjaE3,spvak1k2)
      acd163(53)=abb163(19)
      acd163(54)=dotproduct(ninjaE3,spval5k1)
      acd163(55)=abb163(21)
      acd163(56)=dotproduct(ninjaE3,spvae1k1)
      acd163(57)=abb163(22)
      acd163(58)=dotproduct(ninjaE3,spval5l4)
      acd163(59)=abb163(24)
      acd163(60)=abb163(25)
      acd163(61)=abb163(42)
      acd163(62)=dotproduct(ninjaE3,spvak1e1)
      acd163(63)=abb163(28)
      acd163(64)=dotproduct(ninjaE3,spval5k2)
      acd163(65)=abb163(29)
      acd163(66)=dotproduct(ninjaE3,spval5e2)
      acd163(67)=abb163(31)
      acd163(68)=dotproduct(ninjaE3,spvae1l5)
      acd163(69)=abb163(33)
      acd163(70)=dotproduct(ninjaE3,spval5e1)
      acd163(71)=abb163(46)
      acd163(72)=dotproduct(ninjaE3,spvae1k2)
      acd163(73)=abb163(48)
      acd163(74)=dotproduct(ninjaE3,spvae1l4)
      acd163(75)=abb163(49)
      acd163(76)=dotproduct(ninjaE3,spvae1e2)
      acd163(77)=abb163(52)
      acd163(78)=abb163(54)
      acd163(79)=dotproduct(ninjaE3,spval4e1)
      acd163(80)=abb163(154)
      acd163(81)=dotproduct(ninjaE3,spval4k2)
      acd163(82)=abb163(166)
      acd163(83)=dotproduct(ninjaA,ninjaA)
      acd163(84)=dotproduct(ninjaA,spvae2k2)
      acd163(85)=dotproduct(ninjaA,spvae2e1)
      acd163(86)=dotproduct(ninjaA,spvak2e1)
      acd163(87)=dotproduct(ninjaA,spvak1k2)
      acd163(88)=dotproduct(ninjaA,spval5k1)
      acd163(89)=dotproduct(ninjaA,spvae1k1)
      acd163(90)=dotproduct(ninjaA,spval5l4)
      acd163(91)=dotproduct(ninjaA,spvak1e1)
      acd163(92)=dotproduct(ninjaA,spval5k2)
      acd163(93)=dotproduct(ninjaA,spval5e2)
      acd163(94)=dotproduct(ninjaA,spvae1l5)
      acd163(95)=dotproduct(ninjaA,spval5e1)
      acd163(96)=dotproduct(ninjaA,spvae1k2)
      acd163(97)=dotproduct(ninjaA,spvae1l4)
      acd163(98)=dotproduct(ninjaA,spvae1e2)
      acd163(99)=dotproduct(ninjaA,spval4e1)
      acd163(100)=dotproduct(ninjaA,spval4k2)
      acd163(101)=abb163(13)
      acd163(102)=acd163(22)*acd163(21)
      acd163(103)=acd163(20)*acd163(19)
      acd163(104)=acd163(18)*acd163(17)
      acd163(105)=acd163(16)*acd163(15)
      acd163(106)=acd163(14)*acd163(13)
      acd163(107)=acd163(12)*acd163(11)
      acd163(108)=acd163(10)*acd163(9)
      acd163(109)=acd163(7)*acd163(8)
      acd163(110)=acd163(3)*acd163(4)
      acd163(102)=acd163(106)+acd163(107)+acd163(108)+acd163(109)+acd163(102)+a&
      &cd163(104)+acd163(105)+acd163(103)+acd163(110)
      acd163(102)=acd163(102)*acd163(5)
      acd163(103)=acd163(22)*acd163(29)
      acd163(104)=acd163(20)*acd163(28)
      acd163(105)=acd163(18)*acd163(27)
      acd163(106)=acd163(16)*acd163(26)
      acd163(107)=acd163(14)*acd163(25)
      acd163(108)=acd163(12)*acd163(24)
      acd163(109)=acd163(10)*acd163(23)
      acd163(110)=acd163(7)*acd163(6)
      acd163(111)=acd163(3)*acd163(1)
      acd163(103)=acd163(107)+acd163(106)+acd163(105)+acd163(103)+acd163(104)+a&
      &cd163(108)+acd163(109)+acd163(110)+acd163(111)
      acd163(104)=acd163(103)*acd163(2)
      acd163(102)=acd163(102)+acd163(104)-acd163(30)
      acd163(103)=acd163(32)*acd163(103)
      acd163(104)=acd163(22)*acd163(42)
      acd163(105)=acd163(20)*acd163(41)
      acd163(106)=acd163(18)*acd163(40)
      acd163(107)=acd163(16)*acd163(39)
      acd163(108)=acd163(14)*acd163(38)
      acd163(109)=acd163(12)*acd163(37)
      acd163(110)=acd163(10)*acd163(36)
      acd163(111)=acd163(7)*acd163(34)
      acd163(112)=acd163(3)*acd163(31)
      acd163(104)=acd163(110)+acd163(111)+acd163(112)+acd163(43)+acd163(106)+ac&
      &d163(107)+acd163(108)+acd163(109)+acd163(104)+acd163(105)
      acd163(105)=acd163(5)*acd163(104)
      acd163(106)=-acd163(82)*acd163(81)
      acd163(107)=acd163(80)*acd163(79)
      acd163(108)=acd163(77)*acd163(76)
      acd163(109)=acd163(75)*acd163(74)
      acd163(110)=acd163(73)*acd163(72)
      acd163(111)=acd163(71)*acd163(70)
      acd163(112)=acd163(69)*acd163(68)
      acd163(113)=acd163(67)*acd163(66)
      acd163(114)=acd163(65)*acd163(64)
      acd163(115)=acd163(63)*acd163(62)
      acd163(116)=acd163(59)*acd163(58)
      acd163(117)=acd163(57)*acd163(56)
      acd163(118)=acd163(55)*acd163(54)
      acd163(119)=acd163(53)*acd163(52)
      acd163(120)=acd163(51)*acd163(50)
      acd163(121)=acd163(49)*acd163(48)
      acd163(122)=acd163(47)*acd163(46)
      acd163(123)=acd163(30)*acd163(44)
      acd163(124)=acd163(28)*acd163(78)
      acd163(125)=acd163(27)*acd163(61)
      acd163(126)=acd163(26)*acd163(60)
      acd163(127)=acd163(23)*acd163(45)
      acd163(128)=acd163(6)*acd163(35)
      acd163(129)=acd163(1)*acd163(33)
      acd163(103)=acd163(105)+acd163(103)+acd163(129)+acd163(128)+acd163(127)+a&
      &cd163(126)+acd163(125)+acd163(124)-2.0_ki*acd163(123)+acd163(122)+acd163&
      &(121)+acd163(120)+acd163(119)+acd163(118)+acd163(117)+acd163(116)+acd163&
      &(115)+acd163(114)+acd163(113)+acd163(112)+acd163(111)+acd163(110)+acd163&
      &(109)+acd163(108)+acd163(106)+acd163(107)
      acd163(105)=ninjaP*acd163(102)
      acd163(104)=acd163(32)*acd163(104)
      acd163(106)=-acd163(82)*acd163(100)
      acd163(107)=acd163(80)*acd163(99)
      acd163(108)=acd163(77)*acd163(98)
      acd163(109)=acd163(75)*acd163(97)
      acd163(110)=acd163(73)*acd163(96)
      acd163(111)=acd163(71)*acd163(95)
      acd163(112)=acd163(69)*acd163(94)
      acd163(113)=acd163(67)*acd163(93)
      acd163(114)=acd163(65)*acd163(92)
      acd163(115)=acd163(63)*acd163(91)
      acd163(116)=acd163(59)*acd163(90)
      acd163(117)=acd163(57)*acd163(89)
      acd163(118)=acd163(55)*acd163(88)
      acd163(119)=acd163(53)*acd163(87)
      acd163(120)=acd163(51)*acd163(86)
      acd163(121)=acd163(49)*acd163(85)
      acd163(122)=acd163(47)*acd163(84)
      acd163(123)=acd163(41)*acd163(78)
      acd163(124)=acd163(40)*acd163(61)
      acd163(125)=acd163(39)*acd163(60)
      acd163(126)=acd163(36)*acd163(45)
      acd163(127)=acd163(34)*acd163(35)
      acd163(128)=acd163(31)*acd163(33)
      acd163(129)=-acd163(30)*acd163(83)
      acd163(104)=acd163(104)+acd163(129)+acd163(128)+acd163(127)+acd163(126)+a&
      &cd163(125)+acd163(124)+acd163(123)+acd163(122)+acd163(121)+acd163(120)+a&
      &cd163(119)+acd163(118)+acd163(117)+acd163(116)+acd163(115)+acd163(114)+a&
      &cd163(113)+acd163(112)+acd163(111)+acd163(110)+acd163(109)+acd163(108)+a&
      &cd163(107)+acd163(101)+acd163(106)+acd163(105)
      brack(ninjaidxt1mu0)=acd163(103)
      brack(ninjaidxt0mu0)=acd163(104)
      brack(ninjaidxt0mu2)=acd163(102)
   end subroutine brack_32
!---#] subroutine brack_32:
!---#[ subroutine numerator_t3:
   subroutine numerator_t3(ncut, a, b, c, param, deg, coeffs) &
   & bind(c, name="p2_gg_httbar_d163h12_ninja_t3")
      use iso_c_binding, only: c_int
      use ninjago_module, only: ki => ki_nin
      use p2_gg_httbar_globalsl1, only: epspow
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_abbrevd163h12
      implicit none
      integer(c_int), intent(in) :: ncut, deg
      complex(ki), dimension(0:3), intent(in) :: a, b, c
      complex(ki), intent(in) :: param
      complex(ki), dimension(4) :: vecA, vecB, vecC
      complex(ki), dimension(0:*), intent(out) :: coeffs
      integer :: t1
      complex(ki), dimension(4) :: qshift
      qshift = -k3-k4-k5
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
end module     p2_gg_httbar_d163h12l131
