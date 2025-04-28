module     p2_gg_httbar_d32h12l1_qp
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity12d32h12l1_qp.f90
   ! generator: buildfortran.py
   use p2_gg_httbar_config, only: ki => ki_qp
   use p2_gg_httbar_util_qp, only: cond
   implicit none
   private
   complex(ki), parameter :: i_ = (0.0_ki, 1.0_ki)
   public :: numerator_ninja
contains
!---#[ function brack_1:
   pure function brack_1(Q,mu2) result(brack)
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_color_qp
      use p2_gg_httbar_abbrevd32h12_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki) :: brack
      complex(ki) :: acc32(29)
      complex(ki) :: Qspvae1l4
      complex(ki) :: Qspvae1k2
      complex(ki) :: Qspvae1e2
      complex(ki) :: Qspval3e1
      complex(ki) :: Qspvak2e1
      complex(ki) :: Qspvae2e1
      complex(ki) :: Qspval4e1
      complex(ki) :: Qspvae1l5
      complex(ki) :: Qspvae1l3
      Qspvae1l4 = dotproduct(Q,spvae1l4)
      Qspvae1k2 = dotproduct(Q,spvae1k2)
      Qspvae1e2 = dotproduct(Q,spvae1e2)
      Qspval3e1 = dotproduct(Q,spval3e1)
      Qspvak2e1 = dotproduct(Q,spvak2e1)
      Qspvae2e1 = dotproduct(Q,spvae2e1)
      Qspval4e1 = dotproduct(Q,spval4e1)
      Qspvae1l5 = dotproduct(Q,spvae1l5)
      Qspvae1l3 = dotproduct(Q,spvae1l3)
      acc32(1)=abb32(10)
      acc32(2)=abb32(11)
      acc32(3)=abb32(12)
      acc32(4)=abb32(13)
      acc32(5)=abb32(14)
      acc32(6)=abb32(15)
      acc32(7)=abb32(16)
      acc32(8)=abb32(18)
      acc32(9)=abb32(19)
      acc32(10)=abb32(20)
      acc32(11)=abb32(22)
      acc32(12)=abb32(23)
      acc32(13)=abb32(24)
      acc32(14)=abb32(25)
      acc32(15)=abb32(26)
      acc32(16)=abb32(30)
      acc32(17)=abb32(38)
      acc32(18)=abb32(48)
      acc32(19)=abb32(50)
      acc32(20)=abb32(52)
      acc32(21)=acc32(9)*Qspvae1l4
      acc32(22)=-acc32(10)*Qspvae1k2
      acc32(23)=-acc32(13)*Qspvae1e2
      acc32(21)=acc32(23)+acc32(22)+acc32(21)+acc32(8)
      acc32(21)=Qspval3e1*acc32(21)
      acc32(22)=acc32(4)*Qspvae1k2
      acc32(23)=-acc32(12)*Qspvae1e2
      acc32(24)=acc32(15)*Qspvae1l4
      acc32(22)=acc32(24)+acc32(23)+acc32(6)+acc32(22)
      acc32(22)=Qspvak2e1*acc32(22)
      acc32(23)=acc32(17)*Qspvae2e1
      acc32(24)=acc32(20)*Qspval4e1
      acc32(23)=acc32(24)+acc32(23)+acc32(7)
      acc32(23)=Qspvae1l5*acc32(23)
      acc32(24)=acc32(18)*Qspval4e1
      acc32(25)=acc32(19)*Qspvae2e1
      acc32(24)=acc32(25)+acc32(24)+acc32(3)
      acc32(24)=Qspvae1l3*acc32(24)
      acc32(25)=acc32(1)*Qspvae1k2
      acc32(26)=acc32(5)*Qspvae2e1
      acc32(27)=acc32(11)*Qspvae1e2
      acc32(28)=acc32(14)*Qspvae1l4
      acc32(29)=acc32(16)*Qspval4e1
      brack=acc32(2)+acc32(21)+acc32(22)+acc32(23)+acc32(24)+acc32(25)+acc32(26&
      &)+acc32(27)+acc32(28)+acc32(29)
   end  function brack_1
!---#] function brack_1:
!---#[ numerator interfaces:
   !------#[ subroutine numerator_ninja:
   subroutine numerator_ninja(ncut, Q_ext, mu2_ext, numerator) &
   & bind(c, name="p2_gg_httbar_d32h12l1_qp_ninja")
      use iso_c_binding, only: c_int
      use quadninjago_module, only: ki_nin
      use p2_gg_httbar_globalsl1_qp, only: epspow
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_abbrevd32h12_qp
      implicit none
      integer(c_int), intent(in) :: ncut
      complex(ki_nin), dimension(0:3), intent(in) :: Q_ext
      complex(ki_nin), intent(in) :: mu2_ext
      complex(ki_nin), intent(out) :: numerator
      complex(ki) :: d32
      ! The Q that goes into the diagram
      complex(ki), dimension(4) :: Q
      complex(ki) :: mu2
      real(ki), dimension(0:3) :: qshift
      qshift = k2-k4
      Q(1:4)  =cmplx(real(-Q_ext(0:3)  -qshift(:),  ki_nin), aimag(-Q_ext(0:3))&
      &, ki)
      d32 = 0.0_ki
      d32 = (cond(epspow.eq.0,brack_1,Q,mu2))
      numerator = cmplx(real(d32, ki), aimag(d32), ki_nin)
   end subroutine numerator_ninja
   !------#] subroutine numerator_ninja:
!---#] numerator interfaces:
end module p2_gg_httbar_d32h12l1_qp
