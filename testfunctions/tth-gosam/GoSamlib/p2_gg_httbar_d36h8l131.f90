module     p2_gg_httbar_d36h8l131
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity8d36h8l131.f90
   ! generator: buildfortran_tn3.py
   use p2_gg_httbar_config, only: ki
   use p2_gg_httbar_util, only: cond_t, d => metric_tensor
   implicit none
   private
   complex(ki), parameter :: i_ = (0.0_ki, 1.0_ki)
   integer, parameter :: ninjaidxt3mu0 = 0
   integer, parameter :: ninjaidxt2mu0 = 1
   integer, parameter :: ninjaidxt1mu0 = 2
   integer, parameter :: ninjaidxt1mu2 = 3
   integer, parameter :: ninjaidxt0mu0 = 4
   integer, parameter :: ninjaidxt0mu2 = 5
   public :: numerator_t3
contains
!---#[ subroutine brack_31:
   pure subroutine brack_31(ninjaA, ninjaE3, ninjaE4, ninjaP, brack)
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd36h8
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA, ninjaE3, ninjaE4
      complex(ki), intent(in) :: ninjaP
      complex(ki), dimension(60) :: acd36
      complex(ki), dimension (0:*), intent(inout) :: brack
      acd36(1)=dotproduct(k2,ninjaE3)
      acd36(2)=abb36(16)
      acd36(3)=dotproduct(l3,ninjaE3)
      acd36(4)=abb36(29)
      acd36(5)=dotproduct(l4,ninjaE3)
      acd36(6)=abb36(98)
      acd36(7)=dotproduct(ninjaE3,spval4k2)
      acd36(8)=abb36(15)
      acd36(9)=dotproduct(ninjaE3,spval3k2)
      acd36(10)=abb36(18)
      acd36(11)=dotproduct(ninjaE3,spval3k1)
      acd36(12)=abb36(19)
      acd36(13)=dotproduct(ninjaE3,spvae1k2)
      acd36(14)=abb36(20)
      acd36(15)=dotproduct(ninjaE3,spval4l3)
      acd36(16)=abb36(21)
      acd36(17)=dotproduct(ninjaE3,spval3l4)
      acd36(18)=abb36(22)
      acd36(19)=dotproduct(ninjaE3,spvak2l3)
      acd36(20)=abb36(23)
      acd36(21)=dotproduct(ninjaE3,spval4e2)
      acd36(22)=abb36(24)
      acd36(23)=dotproduct(ninjaE3,spvak1k2)
      acd36(24)=abb36(25)
      acd36(25)=dotproduct(ninjaE3,spval4k1)
      acd36(26)=abb36(26)
      acd36(27)=dotproduct(ninjaE3,spvae2k2)
      acd36(28)=abb36(27)
      acd36(29)=dotproduct(ninjaE3,spval4e1)
      acd36(30)=abb36(30)
      acd36(31)=dotproduct(ninjaE3,spvak1l3)
      acd36(32)=abb36(31)
      acd36(33)=dotproduct(ninjaE3,spvae2l3)
      acd36(34)=abb36(33)
      acd36(35)=dotproduct(ninjaE3,spval3e2)
      acd36(36)=abb36(34)
      acd36(37)=dotproduct(ninjaE3,spvae1l3)
      acd36(38)=abb36(38)
      acd36(39)=dotproduct(ninjaE3,spval3e1)
      acd36(40)=abb36(40)
      acd36(41)=acd36(2)*acd36(1)
      acd36(42)=acd36(4)*acd36(3)
      acd36(43)=acd36(6)*acd36(5)
      acd36(44)=acd36(8)*acd36(7)
      acd36(45)=acd36(10)*acd36(9)
      acd36(46)=acd36(12)*acd36(11)
      acd36(47)=acd36(14)*acd36(13)
      acd36(48)=acd36(16)*acd36(15)
      acd36(49)=acd36(18)*acd36(17)
      acd36(50)=acd36(20)*acd36(19)
      acd36(51)=acd36(22)*acd36(21)
      acd36(52)=acd36(24)*acd36(23)
      acd36(53)=acd36(26)*acd36(25)
      acd36(54)=acd36(28)*acd36(27)
      acd36(55)=acd36(30)*acd36(29)
      acd36(56)=acd36(32)*acd36(31)
      acd36(57)=acd36(34)*acd36(33)
      acd36(58)=acd36(36)*acd36(35)
      acd36(59)=acd36(38)*acd36(37)
      acd36(60)=acd36(40)*acd36(39)
      acd36(41)=acd36(60)+acd36(59)+acd36(58)+acd36(57)+acd36(56)+acd36(55)+acd&
      &36(54)+acd36(53)+acd36(52)+acd36(51)+acd36(50)+acd36(49)+acd36(48)+acd36&
      &(47)+acd36(46)+acd36(45)+acd36(44)+acd36(43)+acd36(41)+acd36(42)
      brack(ninjaidxt3mu0)=0.0_ki
      brack(ninjaidxt2mu0)=acd36(41)
   end subroutine brack_31
!---#] subroutine brack_31:
!---#[ subroutine numerator_t3:
   subroutine numerator_t3(ncut, a, b, c, param, deg, coeffs) &
   & bind(c, name="p2_gg_httbar_d36h8_ninja_t3")
      use iso_c_binding, only: c_int
      use ninjago_module, only: ki => ki_nin
      use p2_gg_httbar_globalsl1, only: epspow
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_abbrevd36h8
      implicit none
      integer(c_int), intent(in) :: ncut, deg
      complex(ki), dimension(0:3), intent(in) :: a, b, c
      complex(ki), intent(in) :: param
      complex(ki), dimension(4) :: vecA, vecB, vecC
      complex(ki), dimension(0:*), intent(out) :: coeffs
      integer :: t1
      vecA(1:4) = - a(0:3)
      vecB(1:4) = - b(0:3)
      vecC(1:4) = - c(0:3)
      if (deg.lt.0) return
      t1 = 0
      call cond_t(epspow.eq.t1,brack_31,vecA,vecB,vecC,param,coeffs)
   end subroutine numerator_t3
!---#] subroutine numerator_t3:
end module     p2_gg_httbar_d36h8l131
