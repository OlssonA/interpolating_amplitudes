module     p2_gg_httbar_d75h8l1_qp
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity8d75h8l1_qp.f90
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
      use p2_gg_httbar_abbrevd75h8_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki) :: brack
      complex(ki) :: acc75(38)
      complex(ki) :: QspQ
      complex(ki) :: Qspvae1l5
      complex(ki) :: Qspvae1e2
      complex(ki) :: Qspvae1l4
      complex(ki) :: Qspvae2k2
      complex(ki) :: Qspval4k2
      complex(ki) :: Qspk2
      complex(ki) :: Qspvak2e1
      complex(ki) :: Qspvae2e1
      complex(ki) :: Qspval4e1
      complex(ki) :: Qspval3e2
      complex(ki) :: Qspval3l5
      complex(ki) :: Qspval3l4
      complex(ki) :: Qspvae1l3
      complex(ki) :: Qspvae2l3
      complex(ki) :: Qspval4l3
      complex(ki) :: Qspvak2l3
      complex(ki) :: Qspval3e1
      complex(ki) :: Qspvak2e2
      complex(ki) :: Qspvak2l5
      complex(ki) :: Qspvak2l4
      complex(ki) :: Qspvae1k2
      QspQ = dotproduct(Q,Q)
      Qspvae1l5 = dotproduct(Q,spvae1l5)
      Qspvae1e2 = dotproduct(Q,spvae1e2)
      Qspvae1l4 = dotproduct(Q,spvae1l4)
      Qspvae2k2 = dotproduct(Q,spvae2k2)
      Qspval4k2 = dotproduct(Q,spval4k2)
      Qspk2 = dotproduct(Q,k2)
      Qspvak2e1 = dotproduct(Q,spvak2e1)
      Qspvae2e1 = dotproduct(Q,spvae2e1)
      Qspval4e1 = dotproduct(Q,spval4e1)
      Qspval3e2 = dotproduct(Q,spval3e2)
      Qspval3l5 = dotproduct(Q,spval3l5)
      Qspval3l4 = dotproduct(Q,spval3l4)
      Qspvae1l3 = dotproduct(Q,spvae1l3)
      Qspvae2l3 = dotproduct(Q,spvae2l3)
      Qspval4l3 = dotproduct(Q,spval4l3)
      Qspvak2l3 = dotproduct(Q,spvak2l3)
      Qspval3e1 = dotproduct(Q,spval3e1)
      Qspvak2e2 = dotproduct(Q,spvak2e2)
      Qspvak2l5 = dotproduct(Q,spvak2l5)
      Qspvak2l4 = dotproduct(Q,spvak2l4)
      Qspvae1k2 = dotproduct(Q,spvae1k2)
      acc75(1)=abb75(9)
      acc75(2)=abb75(10)
      acc75(3)=abb75(11)
      acc75(4)=abb75(12)
      acc75(5)=abb75(13)
      acc75(6)=abb75(14)
      acc75(7)=abb75(15)
      acc75(8)=abb75(16)
      acc75(9)=abb75(17)
      acc75(10)=abb75(18)
      acc75(11)=abb75(19)
      acc75(12)=abb75(20)
      acc75(13)=abb75(21)
      acc75(14)=abb75(22)
      acc75(15)=abb75(23)
      acc75(16)=abb75(24)
      acc75(17)=abb75(25)
      acc75(18)=abb75(26)
      acc75(19)=abb75(27)
      acc75(20)=abb75(28)
      acc75(21)=abb75(29)
      acc75(22)=abb75(30)
      acc75(23)=abb75(31)
      acc75(24)=abb75(32)
      acc75(25)=abb75(33)
      acc75(26)=abb75(34)
      acc75(27)=abb75(35)
      acc75(28)=abb75(39)
      acc75(29)=-acc75(23)*QspQ
      acc75(30)=acc75(11)*Qspvae1l5
      acc75(31)=-acc75(14)*Qspvae1e2
      acc75(32)=-acc75(21)*Qspvae1l4
      acc75(33)=Qspvae2k2*acc75(15)
      acc75(34)=Qspval4k2*acc75(18)
      acc75(35)=Qspk2*acc75(2)
      acc75(29)=acc75(35)+acc75(34)+acc75(33)+acc75(32)+acc75(31)+acc75(30)+acc&
      &75(1)+acc75(29)
      acc75(29)=Qspvak2e1*acc75(29)
      acc75(30)=acc75(3)*Qspvae1l5
      acc75(31)=-acc75(7)*Qspvae1l4
      acc75(32)=-acc75(20)*Qspvae1e2
      acc75(33)=acc75(25)*Qspvae2e1
      acc75(34)=-acc75(28)*Qspval4e1
      acc75(30)=acc75(13)+acc75(34)+acc75(33)+acc75(32)+acc75(30)+acc75(31)
      acc75(30)=QspQ*acc75(30)
      acc75(31)=Qspval3e2*acc75(20)
      acc75(32)=-Qspval3l5*acc75(3)
      acc75(33)=Qspval3l4*acc75(7)
      acc75(31)=acc75(33)+acc75(32)+acc75(31)+acc75(8)
      acc75(31)=Qspvae1l3*acc75(31)
      acc75(32)=-Qspvae2l3*acc75(25)
      acc75(33)=Qspval4l3*acc75(28)
      acc75(34)=Qspvak2l3*acc75(23)
      acc75(32)=acc75(34)+acc75(33)+acc75(32)+acc75(17)
      acc75(32)=Qspval3e1*acc75(32)
      acc75(33)=Qspvak2e2*acc75(24)
      acc75(34)=Qspvak2l5*acc75(26)
      acc75(35)=Qspvak2l4*acc75(27)
      acc75(33)=acc75(35)+acc75(34)+acc75(33)+acc75(9)
      acc75(33)=Qspvae1k2*acc75(33)
      acc75(34)=acc75(6)*Qspvae2e1
      acc75(35)=acc75(22)*Qspval4e1
      acc75(34)=acc75(35)+acc75(34)+acc75(5)
      acc75(34)=Qspvae1l5*acc75(34)
      acc75(35)=acc75(10)*Qspvae1l4
      acc75(36)=acc75(12)*Qspvae1e2
      acc75(37)=acc75(16)*Qspval4e1
      acc75(38)=acc75(19)*Qspvae2e1
      brack=acc75(4)+acc75(29)+acc75(30)+acc75(31)+acc75(32)+acc75(33)+acc75(34&
      &)+acc75(35)+acc75(36)+acc75(37)+acc75(38)
   end  function brack_1
!---#] function brack_1:
!---#[ numerator interfaces:
   !------#[ subroutine numerator_ninja:
   subroutine numerator_ninja(ncut, Q_ext, mu2_ext, numerator) &
   & bind(c, name="p2_gg_httbar_d75h8l1_qp_ninja")
      use iso_c_binding, only: c_int
      use quadninjago_module, only: ki_nin
      use p2_gg_httbar_globalsl1_qp, only: epspow
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_abbrevd75h8_qp
      implicit none
      integer(c_int), intent(in) :: ncut
      complex(ki_nin), dimension(0:3), intent(in) :: Q_ext
      complex(ki_nin), intent(in) :: mu2_ext
      complex(ki_nin), intent(out) :: numerator
      complex(ki) :: d75
      ! The Q that goes into the diagram
      complex(ki), dimension(4) :: Q
      complex(ki) :: mu2
      real(ki), dimension(0:3) :: qshift
      qshift = k2-k3-k4
      Q(1:4)  =cmplx(real(-Q_ext(0:3)  -qshift(:),  ki_nin), aimag(-Q_ext(0:3))&
      &, ki)
      d75 = 0.0_ki
      d75 = (cond(epspow.eq.0,brack_1,Q,mu2))
      numerator = cmplx(real(d75, ki), aimag(d75), ki_nin)
   end subroutine numerator_ninja
   !------#] subroutine numerator_ninja:
!---#] numerator interfaces:
end module p2_gg_httbar_d75h8l1_qp
