module     p2_gg_httbar_d32h4l132
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity4d32h4l132.f90
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
      use p2_gg_httbar_abbrevd32h4
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA0, ninjaA1, ninjaE3, ninja&
      &E4
      complex(ki), dimension(0:2), intent(in) :: ninjaP
      complex(ki), dimension(23) :: acd32
      complex(ki), dimension (0:*), intent(inout) :: brack
      complex(ki) :: ninjaP0, ninjaP1, ninjaP2
      ninjaP0 = ninjaP(0)
      ninjaP1 = ninjaP(1)
      ninjaP2 = ninjaP(2)
      acd32(1)=dotproduct(ninjaE3,spvae1k2)
      acd32(2)=dotproduct(ninjaE3,spval4e1)
      acd32(3)=abb32(11)
      acd32(4)=dotproduct(ninjaE3,spval5e1)
      acd32(5)=abb32(19)
      acd32(6)=dotproduct(ninjaE3,spvae2e1)
      acd32(7)=abb32(20)
      acd32(8)=dotproduct(ninjaE3,spval3e1)
      acd32(9)=abb32(33)
      acd32(10)=dotproduct(ninjaE3,spvae1l3)
      acd32(11)=abb32(46)
      acd32(12)=dotproduct(ninjaE3,spvae1l4)
      acd32(13)=abb32(28)
      acd32(14)=dotproduct(ninjaE3,spvae1e2)
      acd32(15)=abb32(26)
      acd32(16)=abb32(48)
      acd32(17)=abb32(21)
      acd32(18)=abb32(24)
      acd32(19)=-acd32(3)*acd32(2)
      acd32(20)=acd32(5)*acd32(4)
      acd32(21)=-acd32(7)*acd32(6)
      acd32(22)=-acd32(9)*acd32(8)
      acd32(19)=acd32(22)+acd32(21)+acd32(19)+acd32(20)
      acd32(19)=acd32(1)*acd32(19)
      acd32(20)=-acd32(11)*acd32(2)
      acd32(21)=-acd32(16)*acd32(6)
      acd32(20)=acd32(21)+acd32(20)
      acd32(20)=acd32(10)*acd32(20)
      acd32(21)=acd32(13)*acd32(4)
      acd32(22)=acd32(17)*acd32(8)
      acd32(21)=acd32(22)+acd32(21)
      acd32(21)=acd32(12)*acd32(21)
      acd32(22)=-acd32(15)*acd32(4)
      acd32(23)=-acd32(18)*acd32(8)
      acd32(22)=acd32(23)+acd32(22)
      acd32(22)=acd32(14)*acd32(22)
      acd32(19)=acd32(19)+acd32(22)+acd32(21)+acd32(20)
      brack(ninjaidxt1x0mu0)=acd32(19)
   end subroutine brack_21
!---#] subroutine brack_21:
!---#[ subroutine brack_22:
   pure subroutine brack_22(ninjaA0, ninjaA1, ninjaE3, ninjaE4, ninjaP, brack)
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd32h4
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA0, ninjaA1, ninjaE3, ninja&
      &E4
      complex(ki), dimension(0:2), intent(in) :: ninjaP
      complex(ki), dimension(59) :: acd32
      complex(ki), dimension (0:*), intent(inout) :: brack
      complex(ki) :: ninjaP0, ninjaP1, ninjaP2
      ninjaP0 = ninjaP(0)
      ninjaP1 = ninjaP(1)
      ninjaP2 = ninjaP(2)
      acd32(1)=dotproduct(ninjaA1,spvae1k2)
      acd32(2)=dotproduct(ninjaE3,spval4e1)
      acd32(3)=abb32(11)
      acd32(4)=dotproduct(ninjaE3,spvae2e1)
      acd32(5)=abb32(20)
      acd32(6)=dotproduct(ninjaE3,spval3e1)
      acd32(7)=abb32(33)
      acd32(8)=dotproduct(ninjaE3,spval5e1)
      acd32(9)=abb32(19)
      acd32(10)=dotproduct(ninjaA1,spval4e1)
      acd32(11)=dotproduct(ninjaE3,spvae1k2)
      acd32(12)=dotproduct(ninjaE3,spvae1l3)
      acd32(13)=abb32(46)
      acd32(14)=dotproduct(ninjaA1,spvae2e1)
      acd32(15)=abb32(48)
      acd32(16)=dotproduct(ninjaA1,spvae1e2)
      acd32(17)=abb32(24)
      acd32(18)=abb32(26)
      acd32(19)=dotproduct(ninjaA1,spvae1l3)
      acd32(20)=dotproduct(ninjaA1,spval3e1)
      acd32(21)=dotproduct(ninjaE3,spvae1e2)
      acd32(22)=dotproduct(ninjaE3,spvae1l4)
      acd32(23)=abb32(21)
      acd32(24)=dotproduct(ninjaA1,spval5e1)
      acd32(25)=abb32(28)
      acd32(26)=dotproduct(ninjaA1,spvae1l4)
      acd32(27)=dotproduct(ninjaA0,spvae1k2)
      acd32(28)=dotproduct(ninjaA0,spval4e1)
      acd32(29)=dotproduct(ninjaA0,spvae2e1)
      acd32(30)=dotproduct(ninjaA0,spvae1e2)
      acd32(31)=dotproduct(ninjaA0,spvae1l3)
      acd32(32)=dotproduct(ninjaA0,spval3e1)
      acd32(33)=dotproduct(ninjaA0,spval5e1)
      acd32(34)=dotproduct(ninjaA0,spvae1l4)
      acd32(35)=abb32(10)
      acd32(36)=abb32(31)
      acd32(37)=abb32(14)
      acd32(38)=abb32(15)
      acd32(39)=abb32(16)
      acd32(40)=abb32(17)
      acd32(41)=abb32(18)
      acd32(42)=abb32(27)
      acd32(43)=acd32(3)*acd32(2)
      acd32(44)=acd32(5)*acd32(4)
      acd32(45)=acd32(7)*acd32(6)
      acd32(46)=acd32(9)*acd32(8)
      acd32(43)=acd32(43)+acd32(44)+acd32(45)-acd32(46)
      acd32(44)=-acd32(1)*acd32(43)
      acd32(45)=acd32(7)*acd32(11)
      acd32(46)=acd32(17)*acd32(21)
      acd32(47)=acd32(23)*acd32(22)
      acd32(45)=-acd32(47)+acd32(45)+acd32(46)
      acd32(46)=-acd32(20)*acd32(45)
      acd32(47)=acd32(9)*acd32(11)
      acd32(48)=acd32(18)*acd32(21)
      acd32(49)=acd32(25)*acd32(22)
      acd32(47)=acd32(49)+acd32(47)-acd32(48)
      acd32(48)=acd32(24)*acd32(47)
      acd32(49)=acd32(3)*acd32(11)
      acd32(50)=acd32(13)*acd32(12)
      acd32(49)=acd32(49)+acd32(50)
      acd32(50)=-acd32(10)*acd32(49)
      acd32(51)=acd32(5)*acd32(11)
      acd32(52)=acd32(15)*acd32(12)
      acd32(51)=acd32(51)+acd32(52)
      acd32(52)=-acd32(14)*acd32(51)
      acd32(53)=acd32(17)*acd32(6)
      acd32(54)=acd32(18)*acd32(8)
      acd32(53)=acd32(53)+acd32(54)
      acd32(54)=-acd32(16)*acd32(53)
      acd32(55)=acd32(13)*acd32(2)
      acd32(56)=acd32(15)*acd32(4)
      acd32(55)=acd32(55)+acd32(56)
      acd32(56)=-acd32(19)*acd32(55)
      acd32(57)=acd32(23)*acd32(6)
      acd32(58)=acd32(25)*acd32(8)
      acd32(57)=acd32(57)+acd32(58)
      acd32(58)=acd32(26)*acd32(57)
      acd32(44)=acd32(58)+acd32(56)+acd32(54)+acd32(52)+acd32(50)+acd32(48)+acd&
      &32(46)+acd32(44)
      acd32(43)=-acd32(27)*acd32(43)
      acd32(45)=-acd32(32)*acd32(45)
      acd32(46)=acd32(33)*acd32(47)
      acd32(47)=-acd32(28)*acd32(49)
      acd32(48)=-acd32(29)*acd32(51)
      acd32(49)=-acd32(30)*acd32(53)
      acd32(50)=-acd32(31)*acd32(55)
      acd32(51)=acd32(34)*acd32(57)
      acd32(52)=acd32(35)*acd32(11)
      acd32(53)=acd32(36)*acd32(2)
      acd32(54)=acd32(37)*acd32(4)
      acd32(55)=acd32(38)*acd32(21)
      acd32(56)=acd32(39)*acd32(12)
      acd32(57)=acd32(40)*acd32(6)
      acd32(58)=acd32(41)*acd32(8)
      acd32(59)=acd32(42)*acd32(22)
      acd32(43)=acd32(59)+acd32(58)+acd32(57)+acd32(56)+acd32(55)+acd32(54)+acd&
      &32(53)+acd32(52)+acd32(51)+acd32(50)+acd32(49)+acd32(48)+acd32(47)+acd32&
      &(46)+acd32(43)+acd32(45)
      brack(ninjaidxt0x0mu0)=acd32(43)
      brack(ninjaidxt0x1mu0)=acd32(44)
   end subroutine brack_22
!---#] subroutine brack_22:
!---#[ subroutine numerator_t2:
   subroutine numerator_t2(ncut, a0, a1, b, c, param, deg, coeffs) &
   & bind(c, name="p2_gg_httbar_d32h4_ninja_t2")
      use iso_c_binding, only: c_int
      use ninjago_module, only: ki => ki_nin
      use p2_gg_httbar_globalsl1, only: epspow
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_abbrevd32h4
      implicit none
      integer(c_int), intent(in) :: ncut, deg
      complex(ki), dimension(0:3), intent(in) :: a0, a1, b, c
      complex(ki), dimension(0:2), intent(in) :: param
      complex(ki), dimension(4) :: vecA0, vecA1, vecB, vecC
      complex(ki), dimension(0:*), intent(out) :: coeffs
      integer :: t1
      complex(ki), dimension(4) :: qshift
      qshift = k2-k4
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
end module     p2_gg_httbar_d32h4l132
