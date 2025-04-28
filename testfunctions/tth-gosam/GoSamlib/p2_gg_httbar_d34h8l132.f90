module     p2_gg_httbar_d34h8l132
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity8d34h8l132.f90
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
      use p2_gg_httbar_abbrevd34h8
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA0, ninjaA1, ninjaE3, ninja&
      &E4
      complex(ki), dimension(0:2), intent(in) :: ninjaP
      complex(ki), dimension(23) :: acd34
      complex(ki), dimension (0:*), intent(inout) :: brack
      complex(ki) :: ninjaP0, ninjaP1, ninjaP2
      ninjaP0 = ninjaP(0)
      ninjaP1 = ninjaP(1)
      ninjaP2 = ninjaP(2)
      acd34(1)=dotproduct(ninjaE3,spvae1k2)
      acd34(2)=dotproduct(ninjaE3,spval5e1)
      acd34(3)=abb34(11)
      acd34(4)=dotproduct(ninjaE3,spvae2e1)
      acd34(5)=abb34(20)
      acd34(6)=dotproduct(ninjaE3,spval3e1)
      acd34(7)=abb34(33)
      acd34(8)=dotproduct(ninjaE3,spval4e1)
      acd34(9)=abb34(27)
      acd34(10)=dotproduct(ninjaE3,spvae1l3)
      acd34(11)=abb34(46)
      acd34(12)=abb34(48)
      acd34(13)=dotproduct(ninjaE3,spvae1e2)
      acd34(14)=abb34(24)
      acd34(15)=dotproduct(ninjaE3,spvae1l5)
      acd34(16)=abb34(28)
      acd34(17)=abb34(26)
      acd34(18)=abb34(31)
      acd34(19)=acd34(3)*acd34(2)
      acd34(20)=acd34(5)*acd34(4)
      acd34(21)=acd34(7)*acd34(6)
      acd34(22)=-acd34(9)*acd34(8)
      acd34(19)=acd34(22)+acd34(21)+acd34(19)+acd34(20)
      acd34(19)=acd34(1)*acd34(19)
      acd34(20)=acd34(11)*acd34(2)
      acd34(21)=acd34(12)*acd34(4)
      acd34(20)=acd34(21)+acd34(20)
      acd34(20)=acd34(10)*acd34(20)
      acd34(21)=acd34(14)*acd34(6)
      acd34(22)=acd34(17)*acd34(8)
      acd34(21)=acd34(22)+acd34(21)
      acd34(21)=acd34(13)*acd34(21)
      acd34(22)=acd34(16)*acd34(6)
      acd34(23)=acd34(18)*acd34(8)
      acd34(22)=acd34(23)+acd34(22)
      acd34(22)=acd34(15)*acd34(22)
      acd34(19)=acd34(19)+acd34(22)+acd34(21)+acd34(20)
      brack(ninjaidxt1x0mu0)=acd34(19)
   end subroutine brack_21
!---#] subroutine brack_21:
!---#[ subroutine brack_22:
   pure subroutine brack_22(ninjaA0, ninjaA1, ninjaE3, ninjaE4, ninjaP, brack)
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd34h8
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA0, ninjaA1, ninjaE3, ninja&
      &E4
      complex(ki), dimension(0:2), intent(in) :: ninjaP
      complex(ki), dimension(59) :: acd34
      complex(ki), dimension (0:*), intent(inout) :: brack
      complex(ki) :: ninjaP0, ninjaP1, ninjaP2
      ninjaP0 = ninjaP(0)
      ninjaP1 = ninjaP(1)
      ninjaP2 = ninjaP(2)
      acd34(1)=dotproduct(ninjaA1,spvae1k2)
      acd34(2)=dotproduct(ninjaE3,spval5e1)
      acd34(3)=abb34(11)
      acd34(4)=dotproduct(ninjaE3,spvae2e1)
      acd34(5)=abb34(20)
      acd34(6)=dotproduct(ninjaE3,spval3e1)
      acd34(7)=abb34(33)
      acd34(8)=dotproduct(ninjaE3,spval4e1)
      acd34(9)=abb34(27)
      acd34(10)=dotproduct(ninjaA1,spval5e1)
      acd34(11)=dotproduct(ninjaE3,spvae1k2)
      acd34(12)=dotproduct(ninjaE3,spvae1l3)
      acd34(13)=abb34(46)
      acd34(14)=dotproduct(ninjaA1,spvae2e1)
      acd34(15)=abb34(48)
      acd34(16)=dotproduct(ninjaA1,spvae1e2)
      acd34(17)=abb34(24)
      acd34(18)=abb34(26)
      acd34(19)=dotproduct(ninjaA1,spvae1l3)
      acd34(20)=dotproduct(ninjaA1,spvae1l5)
      acd34(21)=abb34(28)
      acd34(22)=abb34(31)
      acd34(23)=dotproduct(ninjaA1,spval3e1)
      acd34(24)=dotproduct(ninjaE3,spvae1e2)
      acd34(25)=dotproduct(ninjaE3,spvae1l5)
      acd34(26)=dotproduct(ninjaA1,spval4e1)
      acd34(27)=dotproduct(ninjaA0,spvae1k2)
      acd34(28)=dotproduct(ninjaA0,spval5e1)
      acd34(29)=dotproduct(ninjaA0,spvae2e1)
      acd34(30)=dotproduct(ninjaA0,spvae1e2)
      acd34(31)=dotproduct(ninjaA0,spvae1l3)
      acd34(32)=dotproduct(ninjaA0,spvae1l5)
      acd34(33)=dotproduct(ninjaA0,spval3e1)
      acd34(34)=dotproduct(ninjaA0,spval4e1)
      acd34(35)=abb34(10)
      acd34(36)=abb34(18)
      acd34(37)=abb34(14)
      acd34(38)=abb34(15)
      acd34(39)=abb34(16)
      acd34(40)=abb34(17)
      acd34(41)=abb34(19)
      acd34(42)=abb34(21)
      acd34(43)=acd34(3)*acd34(2)
      acd34(44)=acd34(5)*acd34(4)
      acd34(45)=acd34(7)*acd34(6)
      acd34(46)=acd34(9)*acd34(8)
      acd34(43)=acd34(43)+acd34(44)+acd34(45)-acd34(46)
      acd34(44)=acd34(1)*acd34(43)
      acd34(45)=acd34(7)*acd34(11)
      acd34(46)=acd34(17)*acd34(24)
      acd34(47)=acd34(21)*acd34(25)
      acd34(45)=acd34(47)+acd34(45)+acd34(46)
      acd34(46)=acd34(23)*acd34(45)
      acd34(47)=acd34(9)*acd34(11)
      acd34(48)=acd34(18)*acd34(24)
      acd34(49)=acd34(22)*acd34(25)
      acd34(47)=-acd34(49)+acd34(47)-acd34(48)
      acd34(48)=-acd34(26)*acd34(47)
      acd34(49)=acd34(3)*acd34(11)
      acd34(50)=acd34(13)*acd34(12)
      acd34(49)=acd34(49)+acd34(50)
      acd34(50)=acd34(10)*acd34(49)
      acd34(51)=acd34(5)*acd34(11)
      acd34(52)=acd34(15)*acd34(12)
      acd34(51)=acd34(51)+acd34(52)
      acd34(52)=acd34(14)*acd34(51)
      acd34(53)=acd34(17)*acd34(6)
      acd34(54)=acd34(18)*acd34(8)
      acd34(53)=acd34(53)+acd34(54)
      acd34(54)=acd34(16)*acd34(53)
      acd34(55)=acd34(13)*acd34(2)
      acd34(56)=acd34(15)*acd34(4)
      acd34(55)=acd34(55)+acd34(56)
      acd34(56)=acd34(19)*acd34(55)
      acd34(57)=acd34(21)*acd34(6)
      acd34(58)=acd34(22)*acd34(8)
      acd34(57)=acd34(57)+acd34(58)
      acd34(58)=acd34(20)*acd34(57)
      acd34(44)=acd34(58)+acd34(56)+acd34(54)+acd34(52)+acd34(50)+acd34(48)+acd&
      &34(46)+acd34(44)
      acd34(43)=acd34(27)*acd34(43)
      acd34(45)=acd34(33)*acd34(45)
      acd34(46)=-acd34(34)*acd34(47)
      acd34(47)=acd34(28)*acd34(49)
      acd34(48)=acd34(29)*acd34(51)
      acd34(49)=acd34(30)*acd34(53)
      acd34(50)=acd34(31)*acd34(55)
      acd34(51)=acd34(32)*acd34(57)
      acd34(52)=acd34(35)*acd34(11)
      acd34(53)=acd34(36)*acd34(2)
      acd34(54)=acd34(37)*acd34(4)
      acd34(55)=acd34(38)*acd34(24)
      acd34(56)=acd34(39)*acd34(12)
      acd34(57)=acd34(40)*acd34(25)
      acd34(58)=acd34(41)*acd34(6)
      acd34(59)=acd34(42)*acd34(8)
      acd34(43)=acd34(59)+acd34(58)+acd34(57)+acd34(56)+acd34(55)+acd34(54)+acd&
      &34(53)+acd34(52)+acd34(51)+acd34(50)+acd34(49)+acd34(48)+acd34(47)+acd34&
      &(46)+acd34(43)+acd34(45)
      brack(ninjaidxt0x0mu0)=acd34(43)
      brack(ninjaidxt0x1mu0)=acd34(44)
   end subroutine brack_22
!---#] subroutine brack_22:
!---#[ subroutine numerator_t2:
   subroutine numerator_t2(ncut, a0, a1, b, c, param, deg, coeffs) &
   & bind(c, name="p2_gg_httbar_d34h8_ninja_t2")
      use iso_c_binding, only: c_int
      use ninjago_module, only: ki => ki_nin
      use p2_gg_httbar_globalsl1, only: epspow
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_abbrevd34h8
      implicit none
      integer(c_int), intent(in) :: ncut, deg
      complex(ki), dimension(0:3), intent(in) :: a0, a1, b, c
      complex(ki), dimension(0:2), intent(in) :: param
      complex(ki), dimension(4) :: vecA0, vecA1, vecB, vecC
      complex(ki), dimension(0:*), intent(out) :: coeffs
      integer :: t1
      complex(ki), dimension(4) :: qshift
      qshift = k2-k5
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
end module     p2_gg_httbar_d34h8l132
