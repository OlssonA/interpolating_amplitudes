module     p2_gg_httbar_d75h12l132_qp
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity12d75h12l132_qp.f90
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
      use p2_gg_httbar_abbrevd75h12_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA0, ninjaA1, ninjaE3, ninja&
      &E4
      complex(ki), dimension(0:2), intent(in) :: ninjaP
      complex(ki), dimension(1) :: acd75
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
      use p2_gg_httbar_abbrevd75h12_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA0, ninjaA1, ninjaE3, ninja&
      &E4
      complex(ki), dimension(0:2), intent(in) :: ninjaP
      complex(ki), dimension(49) :: acd75
      complex(ki), dimension (0:*), intent(inout) :: brack
      complex(ki) :: ninjaP0, ninjaP1, ninjaP2
      ninjaP0 = ninjaP(0)
      ninjaP1 = ninjaP(1)
      ninjaP2 = ninjaP(2)
      acd75(1)=dotproduct(k2,ninjaE3)
      acd75(2)=dotproduct(ninjaE3,spvae1k2)
      acd75(3)=abb75(22)
      acd75(4)=dotproduct(ninjaE3,spvak2e1)
      acd75(5)=abb75(31)
      acd75(6)=dotproduct(ninjaA0,ninjaE3)
      acd75(7)=abb75(21)
      acd75(8)=abb75(18)
      acd75(9)=dotproduct(ninjaE3,spvae1e2)
      acd75(10)=abb75(43)
      acd75(11)=dotproduct(ninjaE3,spvae1l5)
      acd75(12)=abb75(27)
      acd75(13)=dotproduct(ninjaE3,spvae2e1)
      acd75(14)=abb75(39)
      acd75(15)=dotproduct(ninjaE3,spvae1l4)
      acd75(16)=abb75(28)
      acd75(17)=dotproduct(ninjaE3,spval4e1)
      acd75(18)=abb75(29)
      acd75(19)=dotproduct(ninjaE3,spvak2l5)
      acd75(20)=abb75(9)
      acd75(21)=dotproduct(ninjaE3,spvak2l4)
      acd75(22)=abb75(11)
      acd75(23)=abb75(35)
      acd75(24)=dotproduct(ninjaE3,spvak2e2)
      acd75(25)=abb75(37)
      acd75(26)=abb75(15)
      acd75(27)=dotproduct(ninjaE3,spval4k2)
      acd75(28)=abb75(20)
      acd75(29)=dotproduct(ninjaE3,spvae2k2)
      acd75(30)=abb75(24)
      acd75(31)=abb75(25)
      acd75(32)=dotproduct(ninjaE3,spvak2l3)
      acd75(33)=dotproduct(ninjaE3,spval3e1)
      acd75(34)=dotproduct(ninjaE3,spval4l3)
      acd75(35)=dotproduct(ninjaE3,spvae2l3)
      acd75(36)=abb75(19)
      acd75(37)=abb75(32)
      acd75(38)=dotproduct(ninjaE3,spval3k2)
      acd75(39)=dotproduct(ninjaE3,spvae1l3)
      acd75(40)=dotproduct(ninjaE3,spval3l5)
      acd75(41)=dotproduct(ninjaE3,spval3l4)
      acd75(42)=dotproduct(ninjaE3,spval3e2)
      acd75(43)=acd75(7)*acd75(2)
      acd75(44)=acd75(8)*acd75(4)
      acd75(45)=-acd75(10)*acd75(9)
      acd75(46)=acd75(12)*acd75(11)
      acd75(47)=-acd75(14)*acd75(13)
      acd75(48)=acd75(16)*acd75(15)
      acd75(49)=-acd75(18)*acd75(17)
      acd75(43)=acd75(49)+acd75(48)+acd75(47)+acd75(46)+acd75(45)+acd75(43)+acd&
      &75(44)
      acd75(43)=acd75(6)*acd75(43)
      acd75(44)=acd75(5)*acd75(1)
      acd75(45)=acd75(23)*acd75(2)
      acd75(46)=-acd75(26)*acd75(9)
      acd75(47)=acd75(28)*acd75(27)
      acd75(48)=acd75(30)*acd75(29)
      acd75(49)=acd75(31)*acd75(15)
      acd75(44)=acd75(49)+acd75(48)+acd75(47)+acd75(46)+acd75(45)+acd75(44)
      acd75(44)=acd75(4)*acd75(44)
      acd75(45)=acd75(3)*acd75(1)
      acd75(46)=acd75(20)*acd75(19)
      acd75(47)=acd75(22)*acd75(21)
      acd75(48)=acd75(25)*acd75(24)
      acd75(45)=acd75(48)+acd75(47)+acd75(46)+acd75(45)
      acd75(45)=acd75(2)*acd75(45)
      acd75(46)=-acd75(38)*acd75(7)
      acd75(47)=-acd75(40)*acd75(12)
      acd75(48)=-acd75(41)*acd75(16)
      acd75(49)=acd75(42)*acd75(10)
      acd75(46)=acd75(49)+acd75(48)+acd75(47)+acd75(46)
      acd75(46)=acd75(39)*acd75(46)
      acd75(47)=-acd75(32)*acd75(8)
      acd75(48)=acd75(34)*acd75(18)
      acd75(49)=acd75(35)*acd75(14)
      acd75(47)=acd75(49)+acd75(48)+acd75(47)
      acd75(47)=acd75(33)*acd75(47)
      acd75(48)=acd75(36)*acd75(13)
      acd75(49)=acd75(37)*acd75(17)
      acd75(48)=acd75(49)+acd75(48)
      acd75(48)=acd75(11)*acd75(48)
      acd75(43)=2.0_ki*acd75(43)+acd75(44)+acd75(46)+acd75(45)+acd75(47)+acd75(&
      &48)
      brack(ninjaidxt0x0mu0)=acd75(43)
      brack(ninjaidxt0x1mu0)=0.0_ki
   end subroutine brack_22
!---#] subroutine brack_22:
!---#[ subroutine numerator_t2:
   subroutine numerator_t2(ncut, a0, a1, b, c, param, deg, coeffs) &
   & bind(c, name="p2_gg_httbar_d75h12_qp_ninja_t2")
      use iso_c_binding, only: c_int
      use quadninjago_module, only: ki => ki_nin
      use p2_gg_httbar_globalsl1_qp, only: epspow
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_abbrevd75h12_qp
      implicit none
      integer(c_int), intent(in) :: ncut, deg
      complex(ki), dimension(0:3), intent(in) :: a0, a1, b, c
      complex(ki), dimension(0:2), intent(in) :: param
      complex(ki), dimension(4) :: vecA0, vecA1, vecB, vecC
      complex(ki), dimension(0:*), intent(out) :: coeffs
      integer :: t1
      complex(ki), dimension(4) :: qshift
      qshift = k2-k3-k4
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
end module     p2_gg_httbar_d75h12l132_qp
